model-analyzer profile \
  --model-repository=./model_repo \
  --profile-models=resnet50 \
  --output-model-repository-path=results \
  --override-output-model-repository \ # 每次从overrite result目录
  --run-config-search-max-concurrency 2 \
  --run-config-search-max-model-batch-size 2 \
  --run-config-search-max-instance-count 2
