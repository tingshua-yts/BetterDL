#export TF_CPP_MIN_VLOG_LEVEL=2
export TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:2222"], "worker": ["127.0.0.1:3333"], "chief": ["127.0.0.1:4444"]}, "task": {"type": "worker", "index": 0}}'
python ./world_cnn.py