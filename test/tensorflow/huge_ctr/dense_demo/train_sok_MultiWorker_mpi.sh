mpiexec -n 8 --allow-run-as-root \
    python3 huge_ctr/dense_demo/run_sok_MultiWorker_mpi.py \
    --data_filename="./tmp/huge_ctr/256_data_" \
    --global_batch_size=65536 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=100 \
    --nnz_per_slot=10 \
    --num_dense_layers=6 \
    --embedding_vec_size=4 \
    --data_splited=1 \
    --optimizer="adam"