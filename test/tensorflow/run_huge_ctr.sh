#!/bin/bash
set -x
name=huge_ctr

#image=ai-studio-registry.cn-beijing.cr.aliyuncs.com/merlin/merlin-tensorflow-training:22.04
image=nvcr.io/nvidia/merlin/merlin-tensorflow-training:21.09
flag=$(sudo docker ps  | grep "$name" | wc -l)
if [ $flag == 0 ]
then
    sudo nvidia-docker stop "$name"
    sudo nvidia-docker rm "$name"
    sudo nvidia-docker run --name="$name" -d --net=host \
         -v /mnt:/mnt \
    -w /workspace   -it $image /bin/bash
#    -w /workspace   -it pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel /bin/bash
fi
sudo docker exec -it "$name" bash
