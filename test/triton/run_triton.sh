set -x
set -e
nvidia-docker run  --rm --net=host -v /mnt:/mnt nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/mnt/project/BetterDL/test/triton/tmp/triton_deploy_model_repo