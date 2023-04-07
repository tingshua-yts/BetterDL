import tensorflow as tf
import portpicker
import multiprocessing
import os

def create_in_process_cluster(num_workers, num_ps):
    """Creates and starts local servers and returns the cluster_resolver."""
    worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
    ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

    cluster_dict = {}
    cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
    if num_ps > 0:
        cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

    cluster_spec = tf.train.ClusterSpec(cluster_dict)

    # Workers need some inter_ops threads to work properly.
    worker_config = tf.compat.v1.ConfigProto()
    if multiprocessing.cpu_count() < num_workers + 1:
        worker_config.inter_op_parallelism_threads = num_workers + 1

    # 启动worker
    for i in range(num_workers):
        tf.distribute.Server(
            cluster_spec,
            job_name="worker",
            task_index=i,
            config=worker_config,
            protocol="grpc")

    # 启动ps
    for i in range(num_ps):
        tf.distribute.Server(
            cluster_spec,
            job_name="ps",
            task_index=i,
            protocol="grpc")

    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec, rpc_layer="grpc")
    return cluster_resolver

# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

########## 创建ps和worker的process #####################
NUM_WORKERS = 3
NUM_PS = 2
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)

global_batch_size = 64

########### 创建数据 #########################
x = tf.random.uniform((10, 10))
y = tf.random.uniform((10,))

dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
dataset = dataset.batch(global_batch_size)
dataset = dataset.prefetch(2)

########### 设置strategy ##############
variable_partitioner = (
    tf.distribute.experimental.partitioners.MinSizePartitioner(
        min_shard_bytes=(256 << 10),
        max_shards=NUM_PS))

strategy = tf.distribute.ParameterServerStrategy(
    cluster_resolver,
    variable_partitioner=variable_partitioner)

############ 训练 ###################
with strategy.scope():
  model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])

  model.compile(tf.keras.optimizers.legacy.SGD(), loss="mse", steps_per_execution=10)

working_dir = "/tmp/my_working_dir"
log_dir = os.path.join(working_dir, "log")
ckpt_filepath = os.path.join(working_dir, "ckpt")
backup_dir = os.path.join(working_dir, "backup")

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath),
    tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir),
]

model.fit(dataset, epochs=5, steps_per_epoch=20, callbacks=callbacks)