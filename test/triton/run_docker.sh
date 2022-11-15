set -x
name=trition-22.08-py3
image=nvcr.io/nvidia/tritonserver:22.08-py3
flag=$(sudo docker ps  | grep "$name" | wc -l)
if [ $flag == 0 ]
then
    sudo nvidia-docker stop "$name"
    sudo nvidia-docker rm "$name"
    sudo nvidia-docker run --name="$name" -d --net=host \
         -v /mnt:/mnt \
    -w /mnt   -it $image /bin/bash
fi
sudo docker exec -it "$name" bash