export TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:2222"], "worker": ["127.0.0.1:3333", "127.0.0.1:5555"], "chief": ["127.0.0.1:4444"]}, "task": {"type": "worker", "index": 1}}'
#export TF_CPP_MIN_VLOG_LEVEL=2
python ./world_cnn.py