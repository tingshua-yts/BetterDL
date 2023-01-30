def foo(global_batch_size, max_gpu_batch_size, gpus):
    steps = max(1, global_batch_size // (max_gpu_batch_size * gpus)) # must be at least 1
    batch_size = global_batch_size // (steps * gpus)
    true_total = batch_size * steps * gpus
    print(f"global_batch_size: {global_batch_size}\t max_gpu_batch_size:{max_gpu_batch_size}\t gpus:{gpus}\t steps:{steps}\t batch_size:{batch_size}\t true_total:{true_total}")

foo(1024, 256, 4)
foo(1024, 256, 3) # 此时batch size计算出来为341,  超过了max_batch_size
foo(1024, 256, 2)



