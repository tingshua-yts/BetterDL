mpiexec -n 8 --allow-run-as-root \
    python3 ./huge_ctr/dense_demo/run_tf.py \
    --data_filename="./tmp/huge_ctr/256_data_" \
    --global_batch_size=65536 \
    --vocabulary_size=8192 \
    --slot_num=100 \
    --nnz_per_slot=10 \
    --num_dense_layers=6 \
    --embedding_vec_size=4 \
    --optimizer="adam" \
    --data_splited=1