#export TF_CPP_MIN_VLOG_LEVEL=1
export TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:2222"], "chief": ["127.0.0.1:4444"]}, "task": {"type": "chief", "index": 0}}'

python ./train_estimator.py --data_dir=/mnt/data/dbpedia/dbpedia_csv/