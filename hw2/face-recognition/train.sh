#/bin/bash

if [ "" == "${1}" ]; then
    echo 'Please provide node number as argument 1.'
else
    NODE="${1}"

    if [ "1" == "${1}" ]; then
        NODE="01"
    fi

    GPU=false
    GPU_ARG=""
    DOCKER_IMAGE="bvlc/caffe:cpu"

    if type "nvidia-smi" > /dev/null; then
        GPU=true
        GPU_ARG="--gpu=all"
        DOCKER_IMAGE="bvlc/caffe:gpu"
    fi

    CAFFE="caffe train ${GPU_ARG} --solver=\"/hw2/solver-${NODE}.prototxt\" 2>&1 | tee \"${PWD}/log-${NODE}/log.txt\""

    if ! type "docker" > /dev/null; then
        eval "${CAFFE}"
    else
        eval "docker run --rm -v \"${PWD}:/hw2\" ${DOCKER_IMAGE} ${CAFFE}"
    fi
fi
