export TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:2222"], "worker": ["127.0.0.1:3333", "127.0.0.1:4444"]}, "task": {"type": "ps", "index": 0}}'
python ./dist_mnist_train.py