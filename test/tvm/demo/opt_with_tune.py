import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
import tvm.relay as relay
import onnx
import tvm
from tvm.contrib import graph_executor
##########################
### create a TVM runner

number = 2
repeat = 1
min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
timeout = 10  # in seconds

"""
    number: int
        specifies the number of different configurations that we will test.
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int, optional
        specifies how many measurements we will take of each configuration.
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
"""
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)

##########################
### set tuning option
tuning_option = {
    "tuner": "xgb", #use an XGBoost algorithim for guiding the search.

    # For a production job, you will want to set the number of trials to be larger than the value of 20 used here.
    # For CPU we recommend 1500, for GPU 3000-4000
    "trials": 20,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "resnet-50-v2-autotuning.json",
}

##########################
### 加载模型
target = "llvm"
input_name = "data"
shape_dict = {input_name: (1, 3, 244, 244)}
# load onnx model
onnx_model = onnx.load("./tmp/resnet50-new.onnx")
# convert to relay
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)


##########################
### 执行tune
# extracting task,感觉是在创造search space
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Tune the extracted tasks sequentially.
# TODO：本例中tasks的count是25，是怎么确定确定的
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner_obj = XGBTuner(task, loss_type="rank")
    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )


##########################
### compile with tune records
with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

