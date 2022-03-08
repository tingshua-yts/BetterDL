#!/bin/bash
set -x
name=pytorch_docker
flag=$(sudo docker ps  | grep "$name" | wc -l)
if [ $flag == 0 ]
then
    sudo nvidia-docker stop "$name"
    sudo nvidia-docker rm "$name"
    sudo nvidia-docker run --name="$name" -d --net=host \
    -v "$1":/workspace \
    -w /workspace   -it pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel /bin/bash
fi
sudo docker exec -it "$name" bash
