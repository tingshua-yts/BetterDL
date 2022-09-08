 python3 huge_ctr/dense_demo/run_sok_MirroredStrategy.py \
    --data_filename="./tmp/huge_ctr/256_data.file" \
    --global_batch_size=256 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=100 \
    --nnz_per_slot=10 \
    --num_dense_layers=6 \
    --embedding_vec_size=4 \
    --optimizer="adam"