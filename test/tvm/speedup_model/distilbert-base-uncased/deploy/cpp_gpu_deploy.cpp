#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

void DeployGraphExecutor() {
  LOG(INFO) << "Running graph executor...";
  // load in the library
  DLDevice dev{kDLCPU, 0};
  DLDevice gpu_dev{kDLCUDA, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("/mnt/project/BetterDL/test/tvm/tmp/distilbert-base-uncased/tvm/deploy_lib_gpu.so");

  // create the graph executor module
    LOG(INFO) << "create the graph executor module...";

  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(gpu_dev);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  // set input and output mem sapce
  // input tt_a:[[  101  7592  1010  2026  3899  2003 10140  2030  2025   102]], st_a:[[1 1 1 1 1 1 1 1 1 1]]
  // todo 如何从numpy获取结果，然后批量设置
  // todo 在triton的c++ backend中，input获取到的是什么类型的数据结构
  // triton中的 rquest在class InferenceRequest中保存
  LOG(INFO) << "set input and output mem sapce...";
  tvm::runtime::NDArray input_ids = tvm::runtime::NDArray::Empty({1, 10}, DLDataType{kDLInt, 64, 1}, dev);
  tvm::runtime::NDArray mask_attentions = tvm::runtime::NDArray::Empty({1, 10}, DLDataType{kDLInt, 64, 1}, dev);
  tvm::runtime::NDArray output = tvm::runtime::NDArray::Empty({1, 2}, DLDataType{kDLFloat, 32, 1}, dev);

  int input[10] = {101, 7592, 1010, 2026, 3899, 2003, 10140,  2030,  2025,  102};
  for (int i = 0 ; i < 10; ++i) {
      static_cast<long*>(input_ids->data)[i] = input[i];
      static_cast<long*>(mask_attentions->data)[i] = 1;
    }
  // set the right input
  LOG(INFO) << "set the right input...";
  set_input("input_ids", input_ids);
  set_input("attention_mask", mask_attentions);
  // run the code
  LOG(INFO) << "run...";
  for(int i = 0; i < 10000; i++) {
    run();
  }
  LOG(INFO) << "get the output...";
  // get the output
  get_output(0, output);

  for (int i = 0; i < 2; ++i) {
    printf("%f, ", i, static_cast<float*>(output->data)[i]);
  }
}
int main(void) {
  DeployGraphExecutor();
  return 0;
}