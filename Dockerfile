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

# Create directories
RUN mkdir -p VideoPose3D/data/checkpoint
RUN mkdir -p models
RUN mkdir -p input_video

# Download the PoseLifter model
RUN wget -O VideoPose3D/data/checkpoint/pretrained_causal_h36m.bin https://dl.fbaipublicfiles.com/video-pose-3d/cpn-ft-243-dbb-causal.bin

# Download YOLO models
RUN wget -O models/yolov8l.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
RUN wget -O models/yolov8x-pose.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt

# Copy our list of Python packages into the container
COPY requirements.txt .

# --- MODIFIED SECTION ---
# Install gdown first, then install the rest of the requirements.
# gdown is the tool that correctly downloads large files from Google Drive.
RUN pip install --no-cache-dir gdown && pip install --no-cache-dir -r requirements.txt

# Download your video file using gdown.
# YOU MUST REPLACE THE LINK BELOW with your actual Google Drive "Share" link.
RUN gdown 'https://drive.google.com/file/d/1qxnpflDbJiFjKx3GMMeynyKy3clqL7R6/view?usp=sharing' -O input_video/palmer.mp4 && echo "Video download complete. Verifying file size:" && ls -lh input_video/palmer.mp4
# --- END MODIFIED SECTION ---


# Copy only the files that are NOT large
COPY ./homography_data ./homography_data
COPY scoptics_pipeline.py .

# The command that will run automatically when the container starts
CMD ["python", "scoptics_pipeline.py"]