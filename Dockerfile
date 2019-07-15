FROM tensorflow/tensorflow:2.0.0b1-gpu-py3
LABEL maintainer="National Institue of Standards and Technology"

ENV DEBIAN_FRONTEND noninteractive
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

#Create folders
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/images \
    && mkdir -p ${DATA_DIR}/masks \
    && mkdir -p ${DATA_DIR}/outputs

RUN pip3 install lmdb scikit-image

#Copy executable
COPY src ${EXEC_DIR}/

WORKDIR ${EXEC_DIR}

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "train_unet.py"]

