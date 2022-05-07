#!/bin/bash
set -x
name=pytorch_horovod
#image=ai-studio-registry.cn-beijing.cr.aliyuncs.com/tingshuai.yts/horovod:latest
image=registry.cn-hangzhou.aliyuncs.com/ai-samples/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1
flag=$(sudo docker ps  | grep "$name" | wc -l)
if [ $flag == 0 ]
then
    sudo nvidia-docker stop "$name"
    sudo nvidia-docker rm "$name"
    sudo nvidia-docker run --name="$name" -d --net=host \
	 -v "$1":/workspace \
	 -v /mnt/project/BetterDL/test/horovod/.ssh:/root/.ssh \
    -w /workspace   -it $image /bin/bash
fi
sudo docker exec -it "$name" bash
