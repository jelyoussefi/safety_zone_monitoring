#----------------------------------------------------------------------------------------------------------------------
# Flags
#----------------------------------------------------------------------------------------------------------------------
SHELL:=/bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))


#----------------------------------------------------------------------------------------------------------------------
# Docker Settings
#----------------------------------------------------------------------------------------------------------------------
DOCKER_IMAGE_NAME=smart_parking_image
export DOCKER_BUILDKIT=1

MODEL_SIZE ?= m
IMAGE_SIZE ?= 640
MODEL_NAME ?= yolov8${MODEL_SIZE}.pt
INPUT ?= rtsp://admin:admin1234@192.168.1.19:554/h264Preview_01_sub

DEVICE?="CPU"

DOCKER_RUN_PARAMS= \
	-it --rm -a stdout -a stderr -e DISPLAY=${DISPLAY} -e NO_AT_BRIDGE=1  \
	--privileged -v /dev:/dev \
	-p 80:80 \
	-e DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK=0 \
	--cap-add=SYS_ADMIN --cap-add=SYS_PTRACE \
	-v ${CURRENT_DIR}:/workspace \
	-v ${CURRENT_DIR}/.cache:/root/.cache \
	-v /tmp/.X11-unix:/tmp/.X11-unix  -v ${HOME}/.Xauthority:/home/root/.Xauthority \
	${DOCKER_IMAGE_NAME}
	
#----------------------------------------------------------------------------------------------------------------------
# Targets
#----------------------------------------------------------------------------------------------------------------------
default: app
.PHONY:  

build:
	@$(call msg, Building Docker image ${DOCKER_IMAGE_NAME} ...)
	@docker build --rm . -t ${DOCKER_IMAGE_NAME}
	
	
app: build
	@$(call msg, Running the yolov8 demo ...)
	@docker run ${DOCKER_RUN_PARAMS} bash -c ' \
		python3 ./app.py  \
				--detection_model /opt/models/yolo11n/FP16/yolo11n.xml \
				--input ${INPUT} \
				--device ${DEVICE} \
				--config ./configs/config.js \
		'
bash: build
	@docker run ${DOCKER_RUN_PARAMS} bash
#----------------------------------------------------------------------------------------------------------------------
# helper functions
#----------------------------------------------------------------------------------------------------------------------
define msg
	tput setaf 2 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo  "" && \
	echo "         "$1 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo "" && \
	tput sgr0
endef

