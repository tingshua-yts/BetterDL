#export TF_CPP_MIN_VLOG_LEVEL=2

export TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:2222"], "chief": ["127.0.0.1:4444"]}, "task": {"type": "chief", "index": 0}}'
#export TF_CONFIG='{"cluster": {"ps": ["192.168.9.15:2222"], "chief": ["192.168.9.7:4444"]}, "task": {"type": "chief", "index": 0}}'
python ./world_cnn.py