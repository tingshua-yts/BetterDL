import os
import tvm
import time
import itertools
import numpy as np
import tensorflow as tf
from tvm import relay, runtime
from tvm.contrib import graph_executor
from tvm.relay import data_dep_optimization as ddo
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
import scipy.sparse as sp


# Ask tensorflow to limit its GPU memory to what's actually needed
# instead of gobbling everything that's available.
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
# This way this tutorial is a little more friendly to sphinx-gallery.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("tensorflow will use experimental.set_memory_growth(True)")
    except RuntimeError as e:
        print("experimental.set_memory_growth option is not available: {}".format(e))


# The name of the transformer model to download and run.
name = "huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad"
# The number of batches in an input.
batch_size = 1
# The length of each input sequence.
seq_len = 128
# TVM platform identifier. Note that best cpu performance can be achieved by setting -mcpu
# appropriately for your specific machine. CUDA and ROCm are also supported.
target = "llvm"
# Which device to run on. Should be one of tvm.cpu() or tvm.cuda().
dev = tvm.cpu()
# If true, then a sparse variant of the network will be run and
# benchmarked.
measure_sparse = True
# The block size of structured sparsity to convert weight tensors
# into. Changing this parameter may yield speedups for some platforms.
bs_r = 1
# For models besides PruneBert (which is 95% sparse), this parameter
# determines how sparse the generated weights should be. The higher
# the sparsity, the faster the result.
sparsity = 0.85





##############
def load_keras_model(module, name, seq_len, batch_size, report_runtime=True):
    model = module.from_pretrained(name)
    dummy_input = tf.keras.Input(shape=[seq_len], batch_size=batch_size, dtype="int32")
    dummy_out = model(dummy_input)  # Propagate shapes through the keras model.
    if report_runtime:
        np_input = np.random.uniform(size=[batch_size, seq_len], low=0, high=seq_len).astype(
            "int32"
        )
        start = time.time()
        repeats = 50
        for i in range(repeats):
            np_out = model(np_input)
        end = time.time()
        print("Keras Runtime: %f ms." % (1000 * ((end - start) / repeats)))
    return model


def convert_to_graphdef(model, batch_size, seq_len):
    model_func = tf.function(lambda x: model(x))
    input_dict = model._saved_model_inputs_spec
    input_spec = input_dict[list(input_dict.keys())[0]]
    model_func = model_func.get_concrete_function(
        tf.TensorSpec([batch_size, seq_len], input_spec.dtype)
    )
    frozen_func = convert_variables_to_constants_v2(model_func)
    return frozen_func.graph.as_graph_def()


def download_model(name, batch_size, seq_len):
    import transformers

    module = getattr(transformers, "TFBertForSequenceClassification")
    model = load_keras_model(module, name=name, batch_size=batch_size, seq_len=seq_len)
    return convert_to_graphdef(model, batch_size, seq_len)

#############
def import_graphdef(
    name,
    batch_size,
    seq_len,
    save_relay=True,
    relay_file="model.json",
    relay_params="model.params",
):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    shape_dict = {"input_1": (batch_size, seq_len)}
    relay_file = ("%s_%d_%d_%s" % (name, batch_size, seq_len, relay_file)).replace("/", "_")
    relay_params = ("%s_%d_%d_%s" % (name, batch_size, seq_len, relay_params)).replace("/", "_")
    if os.path.exists(os.path.join(abs_path, relay_file)) and os.path.exists(
        os.path.join(abs_path, relay_params)
    ):
        with open(os.path.join(abs_path, relay_file), "r") as fi:
            mod = tvm.ir.load_json(fi.read())
        with open(os.path.join(abs_path, relay_params), "rb") as fi:
            params = relay.load_param_dict(fi.read())
    else:
        graph_def = download_model(name, batch_size, seq_len)

        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict)

        if save_relay:
            with open(os.path.join(abs_path, relay_file), "w") as fo:
                fo.write(tvm.ir.save_json(mod))
            with open(os.path.join(abs_path, relay_params), "wb") as fo:
                fo.write(runtime.save_param_dict(params))

    return mod, dict(params.items()), shape_dict


def run_relay_graph(mod, params, shape_dict, target, dev):
    with relay.build_config(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    input_shape = shape_dict["input_1"]
    dummy_data = np.random.uniform(size=input_shape, low=0, high=input_shape[1]).astype("int32")

    m = graph_executor.GraphModule(lib["default"](dev))
    m.set_input(0, dummy_data)
    m.run()
    tvm_output = m.get_output(0)

    print(m.benchmark(dev, repeat=5, number=5))
    return tvm_output


def run_dense(mod, params, shape_dict, target, dev):
    print("Dense Model Benchmark:")
    return run_relay_graph(mod, params, shape_dict, target, dev)


###
def benchmark():
    mod, params, shape_dict = import_graphdef(name, batch_size, seq_len)
    run_dense(mod, params, shape_dict, target, dev)
    # if measure_sparse:
    #     gen_weights = "prune" not in name
    #     run_sparse(mod, params, shape_dict, target, dev, bs_r, sparsity, gen_weights)