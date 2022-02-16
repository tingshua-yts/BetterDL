set -x
workspace=$1
sudo nvidia-docker stop testcudnn
sudo nvidia-docker rm testcudnn
sudo nvidia-docker run -it --name testcudnn -v $workspace:/workspace nvidia/cuda:10.1-cudnn7-devel-centos7 /bin/bash
sudo nvidia-docker exec -it testcudnn /bin/bash
