# Dockerfile: The official blueprint for running on a powerful NVIDIA GPU in the cloud.

# Start from an official PyTorch image with CUDA support.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# The script needs the 'common' module from the VideoPose3D repository.
RUN git clone https://github.com/facebookresearch/VideoPose3D.git

# --- MODIFIED SECTION: Download models instead of COPYing ---
# Create the directories
RUN mkdir -p VideoPose3D/data/checkpoint
RUN mkdir -p models
RUN mkdir -p input_video

# Download the PoseLifter model
RUN wget -O VideoPose3D/data/checkpoint/pretrained_causal_h36m.bin https://dl.fbaipublicfiles.com/video-pose-3d/cpn-ft-243-dbb-causal.bin

# Download YOLO models directly from Ultralytics
RUN wget -O models/yolov8l.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
RUN wget -O models/yolov8x-pose.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt

# Download your video file. You need to upload this somewhere public like Google Drive or S3.
# For this example, I'll use a placeholder URL.
# YOU MUST REPLACE THIS with a direct download link to your palmer.mp4 video.
RUN wget -O input_video/palmer.mp4 "https://drive.google.com/file/d/1qxnpflDbJiFjKx3GMMeynyKy3clqL7R6/view?usp=sharing"

# --- END MODIFIED SECTION ---

# Copy our list of Python packages into the container
COPY requirements.txt .

# Install all the Python packages from the list
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files that are NOT large
COPY ./homography_data ./homography_data
COPY scoptics_pipeline.py .

# The command that will run automatically when the container starts
CMD ["python", "scoptics_pipeline.py"]