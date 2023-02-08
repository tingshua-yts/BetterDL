from tvm.driver import tvmc
model = tvmc.load('./tmp/resnet50-new.onnx') #Step 1: Load
package = tvmc.compile(model, target="llvm") #Step 2: Compile
result = tvmc.run(package, device="cpu") #Step 3: Run
print(result)
