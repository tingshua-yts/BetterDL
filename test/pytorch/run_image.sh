#!/bin/bash
set -x
name=pytorch_docker2
image=ai-studio-registry.cn-beijing.cr.aliyuncs.com/kube-ai/pytorch:1.11.0-cuda11.3-cudnn8-runtime
flag=$(sudo docker ps  | grep "$name" | wc -l)
if [ $flag == 0 ]
then
    sudo nvidia-docker stop "$name"
    sudo nvidia-docker rm "$name"
    sudo nvidia-docker run --name="$name" -d --net=host \
	 -v "$1":/workspace \
    -w /workspace   -it $image /bin/bash
#    -w /workspace   -it pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel /bin/bash
fi
sudo docker exec -it "$name" bash
