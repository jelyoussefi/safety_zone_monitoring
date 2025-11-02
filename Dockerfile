# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Set non-interactive frontend to avoid prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

#=======================================
# Core System Setup
#=======================================
RUN apt-get update -y && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    gpg \
    pciutils \
    git \
    cmake \
    libglib2.0-0 \
    libtbb12 \
    v4l-utils \
    libusb-1.0-0-dev \
    libssl-dev \
    libgtk-3-dev \
    pkg-config \
    udev \
    libudev-dev \
    python3-pip \
    python3-dev \
    python3-setuptools \
    libopencv-dev \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#=======================================
# Install Intel Graphics Drivers
#=======================================
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get update -y && apt-get install -y \
    libze-intel-gpu1 \
    libze1 \
    intel-metrics-discovery \
    intel-opencl-icd \
    clinfo \
    intel-gsc \
    intel-media-va-driver-non-free \
    libmfx-gen1 \
    libvpl2 \
    libvpl-tools \
    libva-glx2 \
    va-driver-all \
    vainfo \
    libze-dev \
    libzbar0 \
    intel-ocloc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#=======================================
# Install NPU Driver
#=======================================
WORKDIR /tmp
RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-driver-compiler-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-fw-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-level-zero-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    dpkg -i *.deb && \
    rm -f *.deb

#=======================================
# Python Environment
#=======================================
# Install PyTorch and torchvision
RUN pip3 install --break-system-packages \
    torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Install additional dependencies for Python packages
RUN apt-get update -y && apt-get install -y \
    pkg-config \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages, removing python3-blinker if present
RUN apt-get remove -y python3-blinker || true && \
    pip3 install --break-system-packages \
    imutils \
    protobuf \
    Pillow \
    psutils \
    fire \
    distro \
    zeroconf \
    psutil \
    pafy \
    opencv-python \
    psycopg2 \
    screeninfo \
    flask_bootstrap \
    flask \
    flask_restful \
    shapely \
    nncf \
    ultralytics \
    pyzbar

# Install OpenVINO and OpenVINO-dev with ONNX support
RUN pip3 install --break-system-packages --pre -U \
    openvino openvino-dev[onnx] \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

# Set working directory for the application
WORKDIR /workspace
