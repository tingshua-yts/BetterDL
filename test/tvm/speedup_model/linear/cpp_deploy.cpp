#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

void DeployGraphExecutor() {
  LOG(INFO) << "Running graph executor...";
  // load in the library
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("/mnt/project/BetterDL/test/tvm/tmp/linear/tvm/deploy_lib_cpu.so");

  // create the graph executor module
    LOG(INFO) << "create the graph executor module...";

  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  LOG(INFO) << "set input and output mem sapce...";
  tvm::runtime::NDArray input_x = tvm::runtime::NDArray::Empty({5}, DLDataType{kDLInt, 64, 1}, dev);
  tvm::runtime::NDArray input_y = tvm::runtime::NDArray::Empty({5}, DLDataType{kDLInt, 64, 1}, dev);

  tvm::runtime::NDArray output = tvm::runtime::NDArray::Empty({5}, DLDataType{kDLFloat, 32, 1}, dev);


  for (int i = 0 ; i < 5; ++i) {
    static_cast<long*>(input_x->data)[i] = i+1;
    static_cast<long*>(input_y->data)[i] = i+1;

  }

  LOG(INFO) << "set the right input...";
  set_input("x", input_x);
  set_input("y", input_y);

  // run the code
  LOG(INFO) << "run...";
  run();
  LOG(INFO) << "get the output...";
  // get the output
  get_output(0, output);

  for (int i = 0; i < 5; ++i) {
    printf("%f,  ", i, static_cast<float*>(output->data)[i]);
  }
}
int main(void) {
  DeployGraphExecutor();
  return 0;
}