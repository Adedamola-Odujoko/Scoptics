# Dockerfile: ScOptics Backend (GPU-accelerated)

# 1. Base image with PyTorch + CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. Environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

WORKDIR /app

# 3. Install OS dependencies, clone VideoPose3D, create dirs, download weights & YOLO models
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      wget \
      libgl1-mesa-glx \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 \
 # Shallow-clone just the common module
 && git clone --depth=1 https://github.com/facebookresearch/VideoPose3D.git \
 && mkdir -p VideoPose3D/data/checkpoint models \
 && mkdir -p input_video output_video \
 \
 # Download pre-trained pose lifter weights
 && wget -q -O VideoPose3D/data/checkpoint/pretrained_causal_h36m.bin \
       https://dl.fbaipublicfiles.com/video-pose-3d/cpn-ft-243-dbb-causal.bin \
 \
 # Download YOLO detection & pose models
 && wget -q -O models/yolov8l.pt \
       https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt \
 && wget -q -O models/yolov8x-pose.pt \
       https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt

# 4. Copy & install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code & homography data
COPY scoptics_pipeline.py ./
COPY homography_data ./homography_data

# 6. Declare mount points
VOLUME [ "/app/input_video", "/app/output_video" ]

# 7. Default command
CMD ["python", "scoptics_pipeline.py"]
