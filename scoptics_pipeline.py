# =================================================================================
# PIPELINE V36.3 (DEFINITIVE & BATCH-OPTIMIZED - COMPLETE & UN-OMITTED)
# This version integrates high-performance batched inference for the VideoPose3D
# model, significantly increasing processing speed. No sections have been omitted.
# =================================================================================

import json, shutil, numpy as np, torch, time, websocket, cv2, ssl
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
from collections import Counter, deque, defaultdict
from PIL import Image
from sklearn.cluster import KMeans
from typing import Generator, Iterable, List, TypeVar, Dict
import umap
from transformers import AutoProcessor, SiglipVisionModel
from scipy.spatial.transform import Rotation as R, Slerp
import sys, os

# --- Add the cloned repository to our Python path and import the OFFICIAL model class ---
# Ensure this path is correct for your Colab environment
sys.path.append('/app/VideoPose3D')
from common.model import TemporalModel

# --- Initial Setup & Configuration ---
CONFIG = {
    "SOURCE_VIDEO_DRIVE_PATH": "input_video/palmer.mp4",
    "LOCAL_VIDEO_PATH": "input_video/palmer.mp4",
    "CALIBRATION_PATH": "homography_data/palmer_homography.json",
    "OUTPUT_VIDEO_PATH": "output_video/output.mp4",
    "DEVICE": torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    "YOLO_DETECT_MODEL": 'yolov8l.pt',
    "OPTIMIZED_DETECT_MODEL_DRIVE_PATH": "models/yolov8l_detect_960x544_fp16.engine",
    "DETECT_MODEL_INPUT_WIDTH": 960, "DETECT_MODEL_INPUT_HEIGHT": 544,
    "YOLO_POSE_MODEL": 'yolov8x-pose.pt',
    "OPTIMIZED_POSE_MODEL_DRIVE_PATH": "models/yolov8x-pose_640x384_fp16.engine",
    "POSE_MODEL_INPUT_WIDTH": 640, "POSE_MODEL_INPUT_HEIGHT": 384,
    "DETECTION_CONFIDENCE_THRESHOLD": 0.3,
    "POSE_CONFIDENCE_ON_CROP": 0.5,
    "YOLO_PERSON_CLASS_ID": 0,
    "CALIBRATION_INPUT_WIDTH": 960, "CALIBRATION_INPUT_HEIGHT": 540,
    "TEAM_A_LABEL": "Team A", "TEAM_B_LABEL": "Team B", "REFEREE_LABEL": "Referee",
    "CLASSIFIER_INIT_FRAMES": 200,
}
print(f"Using device: {CONFIG['DEVICE']}")
# --- Define Inference Sizes ---
# These variables were removed by accident in the last step. We define them here.
detect_inference_size = [CONFIG['DETECT_MODEL_INPUT_HEIGHT'], CONFIG['DETECT_MODEL_INPUT_WIDTH']]
pose_inference_size = [CONFIG['POSE_MODEL_INPUT_HEIGHT'], CONFIG['POSE_MODEL_INPUT_WIDTH']]

# --- ROBUST I/O ---
if not os.path.exists(CONFIG['SOURCE_VIDEO_DRIVE_PATH']): print(f"FATAL ERROR: Source video not found at {CONFIG['SOURCE_VIDEO_DRIVE_PATH']}"); sys.exit(1)
try:
 # print(f"Copying video to local runtime..."); shutil.copyfile(CONFIG['SOURCE_VIDEO_DRIVE_PATH'], CONFIG['LOCAL_VIDEO_PATH'])
    video_info = sv.VideoInfo.from_video_path(CONFIG['LOCAL_VIDEO_PATH']); print(f"✅ Video validated with {video_info.total_frames} frames.")
except Exception as e: print(f"FATAL ERROR: Could not load video. Reason: {e}"); sys.exit(1)
with open(CONFIG['CALIBRATION_PATH'], 'r') as f: sparse_calibration_data = json.load(f)
print(f"✅ Loaded {len(sparse_calibration_data)} sparse calibration sets.")

# --- Team Classifier Class ---
V = TypeVar("V"); SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'
def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
    batch_size = max(batch_size, 1); current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size: yield current_batch; current_batch = []
        current_batch.append(element)
    if current_batch: yield current_batch
class TeamClassifier:
    def __init__(self, device: str = 'cpu', batch_size: int = 64):
        self.device, self.batch_size = device, batch_size; self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH); self.reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.0)
        self.cluster_model = KMeans(n_clusters=3, n_init='auto', random_state=42); self.cluster_to_team_map = {}; self.is_initialized = False
    def extract_features(self, crops: List[np.ndarray], show_progress: bool = False) -> np.ndarray:
        if not crops: return np.array([])
        crops = [sv.cv2_to_pillow(crop) for crop in crops]; batches = create_batches(crops, self.batch_size); data = []
        iterable = tqdm(batches, desc='Embedding extraction', leave=False) if show_progress else batches
        with torch.no_grad():
            for batch in iterable:
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device); outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy(); data.append(embeddings)
        return np.concatenate(data)
    def fit(self, crops: List[np.ndarray]):
        print("Fitting Team Classifier..."); data = self.extract_features(crops, show_progress=True)
        projections = self.reducer.fit_transform(data); self.cluster_model.fit(projections); cluster_counts = Counter(self.cluster_model.labels_)
        if len(cluster_counts) < 3: print("⚠️ Warning: Found fewer than 3 clusters."); return
        ref_label = min(cluster_counts, key=cluster_counts.get); self.cluster_to_team_map[ref_label] = CONFIG['REFEREE_LABEL']
        team_labels = [l for l in range(3) if l != ref_label]
        self.cluster_to_team_map[team_labels[0]] = CONFIG['TEAM_A_LABEL']; self.cluster_to_team_map[team_labels[1]] = CONFIG['TEAM_B_LABEL']
        self.is_initialized = True; print(f"✅ Team Classifier fit complete. Mapping: {self.cluster_to_team_map}")
    def predict_teams(self, crops: List[np.ndarray]) -> List[str]:
        if not self.is_initialized or len(crops) == 0: return ["Unknown"] * len(crops)
        data = self.extract_features(crops, show_progress=False); projections = self.reducer.transform(data)
        cluster_labels = self.cluster_model.predict(projections)
        return [self.cluster_to_team_map.get(label, "Unknown") for label in cluster_labels]

# --- Batched PoseLifter Class for High Performance ---
class BatchedPoseLifter:
    def __init__(self, device):
        self.device = device
        self.model = TemporalModel(num_joints_in=17, in_features=2, num_joints_out=17,
                                   filter_widths=[3, 3, 3, 3, 3], causal=True, dropout=0.25, channels=1024).to(device)
        model_path = "VideoPose3D/data/checkpoint/pretrained_causal_h36m.bin"
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_pos'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("✅ Batched VideoPose3D model loaded.")
        self.receptive_field = self.model.receptive_field()
        self._buffers = defaultdict(lambda: deque(maxlen=self.receptive_field))

    @torch.no_grad()
    def lift_batch(self, keypoints_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        batch_tids = []
        batch_sequences = []

        for tid, keypoints_2d in keypoints_dict.items():
            if keypoints_2d.shape != (17, 2): continue
            keypoints_normalized = keypoints_2d - keypoints_2d[0]
            self._buffers[tid].append(keypoints_normalized)
            if len(self._buffers[tid]) == self.receptive_field:
                batch_tids.append(tid)
                batch_sequences.append(list(self._buffers[tid]))

        if not batch_sequences:
            return {}

        input_tensor = torch.tensor(batch_sequences, dtype=torch.float32, device=self.device)
        output_batch = self.model(input_tensor)
        output_poses = output_batch[:, -1, :, :].cpu().numpy()

        return {tid: pose for tid, pose in zip(batch_tids, output_poses)}

# --- Helper Functions ---
def smooth_camera_parameters(sparse_data, total_frames):
    print("🚀 Starting camera parameter smoothing..."); dense_data = {}; anchor_frames = sorted([int(k) for k in sparse_data.keys()])
    if not anchor_frames: return {}
    for frame_idx in anchor_frames: dense_data[frame_idx] = sparse_data[str(frame_idx)]['cam_params']
    for i in range(len(anchor_frames) - 1):
        start_frame, end_frame = anchor_frames[i], anchor_frames[i+1]; start_params, end_params = dense_data[start_frame], dense_data[end_frame]
        if end_frame - start_frame <= 1: continue
        start_pos, end_pos = np.array(start_params['position_meters']), np.array(end_params['position_meters']); slerp = Slerp([start_frame, end_frame], R.from_matrix([start_params['rotation_matrix'], end_params['rotation_matrix']]))
        for frame_idx in range(start_frame + 1, end_frame):
            alpha = (frame_idx - start_frame) / (end_frame - start_frame); dense_data[frame_idx] = {'position_meters':((1-alpha)*start_pos+alpha*end_pos).tolist(),'rotation_matrix':slerp([frame_idx]).as_matrix()[0].tolist(),'x_focal_length':(1-alpha)*start_params['x_focal_length']+alpha*end_params['x_focal_length'],'y_focal_length':(1-alpha)*start_params['y_focal_length']+alpha*end_params['y_focal_length'],'principal_point':start_params['principal_point']}
    first_anchor, last_anchor = anchor_frames[0], anchor_frames[-1]
    for frame_idx in range(first_anchor): dense_data[frame_idx] = dense_data[first_anchor]
    for frame_idx in range(last_anchor + 1, total_frames): dense_data[frame_idx] = dense_data[last_anchor]
    print(f"✅ Smoothing complete."); return dense_data

def get_projection_params(cam_params):
    if cam_params is None: return None, None, None
    try:
        K = np.array([[cam_params['x_focal_length'], 0, cam_params['principal_point'][0]],[0, cam_params['y_focal_length'], cam_params['principal_point'][1]],[0, 0, 1]]); K_inv = np.linalg.inv(K)
        R_T = np.array(cam_params['rotation_matrix']).T; C = np.array(cam_params['position_meters']).reshape(3, 1); return K_inv, R_T, C
    except (np.linalg.LinAlgError, KeyError): return None, None, None

def project_point(image_point, K_inv, R_T, C):
    ray_cam = K_inv @ np.array([image_point[0], image_point[1], 1.0]); ray_world = R_T @ ray_cam
    if ray_world[2] != 0: s = -C[2] / ray_world[2]; return (C + s * ray_world.reshape(3,1))[:2].ravel() if s > 0 else None
    return None

def remap_coco_to_h36m(coco_keypoints: np.ndarray) -> np.ndarray:
    if coco_keypoints.shape != (17, 2): return None
    h36m_keypoints = np.zeros((17, 2), dtype=np.float32)
    h36m_keypoints[0] = (coco_keypoints[11] + coco_keypoints[12]) / 2
    h36m_keypoints[1] = coco_keypoints[12]; h36m_keypoints[2] = coco_keypoints[14]; h36m_keypoints[3] = coco_keypoints[16]
    h36m_keypoints[4] = coco_keypoints[11]; h36m_keypoints[5] = coco_keypoints[13]; h36m_keypoints[6] = coco_keypoints[15]
    shoulder_midpoint = (coco_keypoints[5] + coco_keypoints[6]) / 2
    h36m_keypoints[7] = shoulder_midpoint; h36m_keypoints[9] = coco_keypoints[0]
    h36m_keypoints[8] = shoulder_midpoint + (coco_keypoints[0] - shoulder_midpoint) * 0.5
    h36m_keypoints[10] = coco_keypoints[0] + (coco_keypoints[0] - shoulder_midpoint) * 0.5
    h36m_keypoints[11] = coco_keypoints[5]; h36m_keypoints[12] = coco_keypoints[7]; h36m_keypoints[13] = coco_keypoints[9]
    h36m_keypoints[14] = coco_keypoints[6]; h36m_keypoints[15] = coco_keypoints[8]; h36m_keypoints[16] = coco_keypoints[10]
    return h36m_keypoints


# --- MAIN EXECUTION ---
# --- MAIN EXECUTION ---
print("\nStarting Main Pipeline (Batched Pose Estimation)...")

# --- Smart Model Loading v3: On-Demand TensorRT Export ---
# This logic checks if an optimized .engine file exists.
# If it doesn't, it builds one from the base .pt file.
# This ensures the .engine file is perfectly compatible with the host GPU.

# --- Smart Model Loading v6: Final, Validated Self-Healing TensorRT Export ---
# This version validates the engine by performing a dummy inference run.
# This is the only definitive way to know if a TensorRT engine is compatible.

def get_or_build_yolo_model(base_model_path, engine_path, inference_size, task):
    # First, try to load and validate the engine file if it exists.
    if os.path.exists(engine_path):
        print(f"✅ Optimized '{task}' model found. Validating compatibility...")
        try:
            model = YOLO(engine_path, task=task)
            
            # Create a dummy black image for the validation run.
            dummy_input = np.zeros((inference_size[0], inference_size[1], 3), dtype=np.uint8)
            
            # The real test: run a single prediction. If this fails, the engine is bad.
            model(dummy_input, verbose=False)
            
            print(f"   Validation successful! Model is compatible.")
            return model
        except Exception as e:
            print(f"❌ FAILED to validate the loaded model from {engine_path}. Reason: {e}")
            print(f"   The existing .engine file is incompatible. Deleting and rebuilding...")
            os.remove(engine_path) # Delete the bad engine file

    # If the engine file didn't exist or failed validation, we build it.
    print(f"⚠️ Building new optimized '{task}' model from base: {base_model_path}...")
    
    if not os.path.exists(base_model_path):
        print(f"FATAL: Base model '{base_model_path}' not found. Cannot build engine.")
        sys.exit(1)
        
    model = YOLO(base_model_path, task=task)
    model.export(format='tensorrt', half=True, workspace=8, imgsz=inference_size)
    
    # Find the exported file and move it to the correct path
    exported_file = base_model_path.replace('.pt', '.engine')
    shutil.move(exported_file, engine_path)
    
    print(f"✅ Successfully built and saved new optimized model to: {engine_path}")
    return YOLO(engine_path, task=task)
    model.export(format='tensorrt', half=True, workspace=8, imgsz=inference_size)
    
    # Find the exported file and move it to the correct path
    exported_file = base_model_path.replace('.pt', '.engine')
    shutil.move(exported_file, engine_path)
    
    print(f"✅ Successfully built and saved new optimized model to: {engine_path}")
    return YOLO(engine_path, task=task)

# Define paths and sizes
detect_engine_path = CONFIG['OPTIMIZED_DETECT_MODEL_DRIVE_PATH']
detect_pt_path = 'models/yolov8l.pt' # Ensure this file is in your models folder
detect_inference_size = [CONFIG['DETECT_MODEL_INPUT_HEIGHT'], CONFIG['DETECT_MODEL_INPUT_WIDTH']]

pose_engine_path = CONFIG['OPTIMIZED_POSE_MODEL_DRIVE_PATH']
pose_pt_path = 'models/yolov8x-pose.pt' # Ensure this file is in your models folder
pose_inference_size = [CONFIG['POSE_MODEL_INPUT_HEIGHT'], CONFIG['POSE_MODEL_INPUT_WIDTH']]

# Get or build the models
detect_detector = get_or_build_yolo_model(detect_pt_path, detect_engine_path, detect_inference_size, 'detect')
pose_detector = get_or_build_yolo_model(pose_pt_path, pose_engine_path, pose_inference_size, 'pose')

print("✅ All models loaded.")

# --- STAGE 1: PRE-COMPUTATION FOR TEAM CLASSIFIER ---
print("--- Stage 1: Collecting crops and fitting classifier ---")
team_classifier = TeamClassifier(device=CONFIG['DEVICE'])
fit_frame_generator = sv.get_video_frames_generator(source_path=CONFIG['LOCAL_VIDEO_PATH'], stride=15)
training_crops = []
for frame in tqdm(fit_frame_generator, desc="Collecting training crops"):
    results = detect_detector(frame, imgsz=detect_inference_size, conf=CONFIG['DETECTION_CONFIDENCE_THRESHOLD'], verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    person_detections = detections[detections.class_id == CONFIG['YOLO_PERSON_CLASS_ID']]
    training_crops.extend([sv.crop_image(frame, xyxy) for xyxy in person_detections.xyxy])
if training_crops: team_classifier.fit(training_crops)
else: print("⚠️ No players found in video sample. Cannot fit classifier.")

# --- STAGE 2: REAL-TIME PROCESSING ---
print("\n--- Stage 2: Starting real-time tracking and prediction ---")
dense_calibration_data = smooth_camera_parameters(sparse_calibration_data, video_info.total_frames)
tracker = sv.ByteTrack(frame_rate=video_info.fps, lost_track_buffer=90)
team_assignments = {}
last_known_keypoints = {}
last_known_poses_3d = {}
w_orig, h_orig = video_info.width, video_info.height
w_calib_scale, h_calib_scale = CONFIG["CALIBRATION_INPUT_WIDTH"]/w_orig, CONFIG["CALIBRATION_INPUT_HEIGHT"]/h_orig
frame_generator = sv.get_video_frames_generator(source_path=CONFIG['LOCAL_VIDEO_PATH'])
pose_lifter = BatchedPoseLifter(device=CONFIG['DEVICE'])
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK, text_color=sv.Color.WHITE)

ws = None
ws_url = "wss://8649e45c8a00.ngrok-free.app" # Your ngrok URL here

def send_data_resiliently(frame_data):
    global ws
    try:
        if not ws or not ws.connected:
            print("🔌 WebSocket not connected. Attempting to connect...")
            ws = websocket.create_connection(ws_url)
            print("✅ WebSocket connected.")
        ws.send(json.dumps(frame_data))
    except (websocket.WebSocketConnectionClosedException, ConnectionResetError, BrokenPipeError, ssl.SSLError, OSError) as e:
        print(f"❗️ WebSocket connection error: {e}. Reconnecting on next frame.")
        if ws: ws.close()
        ws = None
    except Exception as e:
        print(f"🔥 An unexpected WebSocket error occurred: {e}")
        if ws: ws.close()
        ws = None

with sv.VideoSink(CONFIG['OUTPUT_VIDEO_PATH'], video_info) as sink:
    pbar = tqdm(total=video_info.total_frames, desc="Processing Pipeline")
    for frame_num, frame in enumerate(frame_generator):
        detect_results = detect_detector(frame, imgsz=detect_inference_size, conf=CONFIG['DETECTION_CONFIDENCE_THRESHOLD'], verbose=False)[0]
        person_detections = sv.Detections.from_ultralytics(detect_results)
        person_detections = person_detections[person_detections.class_id == CONFIG['YOLO_PERSON_CLASS_ID']]
        tracked_detections = tracker.update_with_detections(person_detections)

        tid_to_keypoints_this_frame = {}
        if len(tracked_detections) > 0:
            for i in range(len(tracked_detections.tracker_id)):
                tid, player_box = tracked_detections.tracker_id[i], tracked_detections.xyxy[i]
                player_crop = sv.crop_image(frame, player_box)
                if player_crop.size == 0: continue
                pose_result = pose_detector(player_crop, imgsz=pose_inference_size, conf=CONFIG['POSE_CONFIDENCE_ON_CROP'], verbose=False)[0]
                if pose_result.keypoints and len(pose_result.keypoints) > 0:
                    keypoints_np = pose_result.keypoints[0].xy.cpu().numpy().squeeze()
                    if keypoints_np.ndim == 2 and keypoints_np.shape == (17, 2):
                        keypoints_np[:, 0] += player_box[0]; keypoints_np[:, 1] += player_box[1]
                        last_known_keypoints[tid] = keypoints_np
                        tid_to_keypoints_this_frame[tid] = keypoints_np

        if team_classifier.is_initialized and len(tracked_detections) > 0:
            new_tids = [tid for tid in tracked_detections.tracker_id if tid not in team_assignments]
            if new_tids:
                new_crops = [sv.crop_image(frame, tracked_detections[np.isin(tracked_detections.tracker_id, tid)].xyxy[0]) for tid in new_tids]
                if new_crops:
                    predicted_teams = team_classifier.predict_teams(new_crops)
                    for tid, team in zip(new_tids, predicted_teams): team_assignments[tid] = team

        poses_3d = {}
        keypoints_to_lift_batch = {}
        for tid in tracked_detections.tracker_id:
            keypoints = tid_to_keypoints_this_frame.get(tid, last_known_keypoints.get(tid))
            if keypoints is not None:
                h36m_keypoints = remap_coco_to_h36m(keypoints)
                if h36m_keypoints is not None:
                    keypoints_to_lift_batch[tid] = h36m_keypoints

        if keypoints_to_lift_batch:
            poses_3d_batch = pose_lifter.lift_batch(keypoints_to_lift_batch)
            for tid, pose in poses_3d_batch.items():
                last_known_poses_3d[tid] = pose.tolist()

        for tid in tracked_detections.tracker_id:
            if tid in last_known_poses_3d:
                poses_3d[tid] = last_known_poses_3d[tid]

        projected_positions = {}
        current_cam_params = dense_calibration_data.get(frame_num)
        if current_cam_params:
            K_inv, R_T, C = get_projection_params(current_cam_params)
            if K_inv is not None:
                for i, tid in enumerate(tracked_detections.tracker_id):
                    box = tracked_detections.xyxy[i]
                    foot_point = np.array([(box[0] + box[2]) / 2 * w_calib_scale, box[3] * h_calib_scale])
                    if (pos := project_point(foot_point, K_inv, R_T, C)) is not None:
                        projected_positions[tid] = pos

        players_data = []
        for tid, pos in projected_positions.items():
            if np.all(np.isfinite(pos)):
                player_data = {"id": int(tid), "x": int(pos[0]*100), "y": int(pos[1]*100),
                               "team": team_assignments.get(tid, "Unknown"), "pose3d": poses_3d.get(tid)}
                players_data.append(player_data)
        frame_data = {"players": players_data}
        send_data_resiliently(frame_data)

        annotated_frame = frame.copy()
        labels = [f"ID:{tid} ({team_assignments.get(tid,'?')})" for tid in tracked_detections.tracker_id]
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
        for tid in tracked_detections.tracker_id:
            keypoints_to_draw, color = last_known_keypoints.get(tid), (0, 0, 255)
            if tid in tid_to_keypoints_this_frame: color = (0, 255, 0)
            if keypoints_to_draw is not None:
                for point in keypoints_to_draw:
                    center = (int(point[0]), int(point[1]))
                    cv2.circle(annotated_frame, center, radius=3, color=color, thickness=-1)

        sink.write_frame(annotated_frame)
        pbar.update(1)

pbar.close()
if ws and ws.connected:
    ws.close()
    print("🔌 Final WebSocket connection closed.")
print("✅ Pipeline finished.")