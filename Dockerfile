# Dockerfile: The official blueprint for running on a powerful NVIDIA GPU in the cloud.

# Start from an official PyTorch image with CUDA support.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by our Python libraries and tools
# Set a non-interactive frontend for package installers to prevent them from hanging.
ENV DEBIAN_FRONTEND=noninteractive
# Set a default timezone (UTC is standard for servers) to prevent tzdata prompts.
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# The script needs the 'common' module from the VideoPose3D repository.
# We clone the entire repository into our container to make the import work.
RUN git clone https://github.com/facebookresearch/VideoPose3D.git

# The script also needs the pre-trained model file for the PoseLifter.
# We create the directory for it and download the file directly into our image.
RUN mkdir -p VideoPose3D/data/checkpoint
RUN wget -O VideoPose3D/data/checkpoint/pretrained_causal_h36m.bin https://dl.fbaipublicfiles.com/video-pose-3d/cpn-ft-243-dbb-causal.bin

# Copy our list of Python packages into the container
COPY requirements.txt .

# Install all the Python packages from the list
RUN pip install --no-cache-dir -r requirements.txt

# Copy all of our project's data and code from your Mac into the container
COPY ./homography_data ./homography_data
COPY ./models ./models
COPY ./input_video ./input_video
COPY scoptics_pipeline.py .

# The command that will run automatically when the container starts
CMD ["python", "scoptics_pipeline.py"]