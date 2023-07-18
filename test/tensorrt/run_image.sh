#!/bin/bash
set -x
name=trt
image=nvcr.io/nvidia/tensorrt:23.06-py3
workspace=`pwd`
flag=$(sudo docker ps  | grep "$name" | wc -l)
if [ $flag == 0 ]
then
    sudo nvidia-docker stop "$name"
    sudo nvidia-docker rm "$name"
    sudo nvidia-docker run --name="$name" -d --network=host \
	 -v /mnt:/mnt \
	 --ipc=host \
    -w $workspace   -it $image /bin/bash
fi
sudo docker exec -it "$name" bash
