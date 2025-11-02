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

PERSON_DET_MODEL ?= yolo11n.pt
HELMET_DET_MODEL ?= ./models/yolo11s_helmet.pt
QR_CODE_DET_MODEL ?= ./models/yolo11s_qr_code.pt

INPUT=./videos/video.mp4

DEVICE?=GPU

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
	@$(call msg, Running the Safety Zone Monitoring ...)
	@docker run ${DOCKER_RUN_PARAMS} bash -c ' \
		python3 ./app.py  \
				--debug_qr=False	 \
				--det_person ${PERSON_DET_MODEL} \
				--det_helmet ${HELMET_DET_MODEL} \
				--det_qr_code ${QR_CODE_DET_MODEL} \
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

