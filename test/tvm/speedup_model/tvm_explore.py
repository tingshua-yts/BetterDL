from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import torch
import torchvision
import time
import pandas as pd
from scipy.special import softmax
import random
import torchvision.transforms as transforms
import glob
import tqdm
import argparse

import os
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"
os.environ["TVM_HOME"] = "~/CodeBase/General/tvm"

class AnimalClassification():
    def __init__(self, testbatch, img_shape=224, trainbatch=16) -> None:
        self.width = img_shape
        self.height = img_shape
        self.outfeatures = 3

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.my_preprocess = transforms.Compose(
            [
                transforms.Resize(self.height),
                transforms.CenterCrop(self.height),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.my_preprocess2 = transforms.Compose(
            [
                transforms.Resize(self.height),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dt_training_path = os.path.join(CommonData().curdir_path, '..', 'Dataset', 'training')
        self.traindata = CustomAnimalClassLoader(dt_training_path)
        self.trainloader = torch.utils.data.DataLoader(self.traindata, batch_size=trainbatch,
                                                shuffle=False, num_workers=10)

        dt_validation_path = os.path.join(CommonData().curdir_path, '..', 'Dataset', 'validation')
        self.testdata = CustomAnimalClassLoader(dt_validation_path)
        self.testloader = torch.utils.data.DataLoader(self.testdata, batch_size=testbatch,
                                                shuffle=False, num_workers=10)

class CustomAnimalClassLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_shape=224):
        super().__init__()
        self.img_dir = img_dir

        self.images = []
        self.labels = []
        self.img_shape = img_shape

        self.class_map = {0:'Cheetah', 1:'Hyena', 2:'Tiger'}
        for i, cl in self.class_map.items():
            each_path = os.path.join(self.img_dir, cl, '*')
            te_img = glob.glob(each_path, recursive=False)
            self.images.extend(te_img)

            le_img = list(np.ones(len(te_img)) * i)
            self.labels.extend(le_img)
            print(f"Found {len(te_img)} images in class {cl}")

        for i in range(len(self.images)):
            if not self.class_map[self.labels[i]] in self.images[i]:
                print(f"{i} : {self.labels[i]}  {self.class_map[self.labels[i]]}   {self.images[i]}")
                raise "error"

        random.seed(100)
        random.shuffle(self.images)
        random.seed(100)
        random.shuffle(self.labels)

        for i in range(len(self.images)):
            if not self.class_map[self.labels[i]] in self.images[i]:
                print(f"{i} : {self.labels[i]}  {self.class_map[self.labels[i]]}   {self.images[i]}")
                raise "error"

        assert(len(self.images)==len(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        my_preprocess2 = transforms.Compose(
                    [
                        transforms.Resize(self.img_shape),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )

        pil_img = Image.open(self.images[index])
        img = my_preprocess2(pil_img)
        lbl = torch.tensor(self.labels[index], dtype=torch.long)
        filename = self.images[index]

        return img, lbl, filename

class CommonData():
    def __init__(self, batch=8, model_name='resnet50') -> None:
        self.batch = batch
        self.model_name = model_name
        self.curdir_path = os.path.dirname(os.path.realpath(__file__))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = os.path.join(self.curdir_path, 'saved_model')
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)

    def warmup(self, model):
        print("torch model warmup stage...")
        random_gen_img = torch.rand(self.batch, 3, 224, 224)
        random_gen_img =  random_gen_img.to(self.device)
        warmup_itr = 5
        for _ in range(warmup_itr):
            model(random_gen_img)

        return model

    def tvm_warmup(self, model):
        print("tvm model warmup stage...")
        random_gen_img = torch.rand(self.batch, 3, 224, 224)
        #random_gen_img =  random_gen_img.to(device)
        warmup_itr = 5
        for _ in range(warmup_itr):
            model.set_input("data", tvm.nd.array(random_gen_img.numpy()))
            model.run()

        return model

    def get_stats(self, csv_file):
        trace_d = pd.read_csv(csv_file)
        trace_d.drop(axis=1, inplace=True, index=0)
        time_columns = [col for col in trace_d.columns if 'time' in col]
        mean_time = trace_d[time_columns].mean().mean()
        print(f"Inference took {mean_time}ms")

def run_pytorch_inference(common_obj:CommonData):
    print("\n#### PYTORCH Inference start ####")
    model = torchvision.models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    for layer in model.parameters():
        layer.requires_grad = False

    model.fc = torch.nn.Linear(in_features=2048, out_features=3, bias=True)
    model = model.to(common_obj.device)
    model.eval()
    csv_name = f"TVM_disabled_{common_obj.model_name}_bch{common_obj.batch}.csv"
    inference_time = []
    predictions = []
    filename = []
    results = pd.DataFrame()

    #warmup stage
    model = common_obj.warmup(model)

    dls = AnimalClassification(common_obj.batch)
    count = 0
    for data, labels_, files in tqdm.tqdm(dls.testloader):
        data = data.to(common_obj.device)
        labels_ = labels_.to(common_obj.device)
        if(data.shape[0] != common_obj.batch):
            print("Ignored bath size ", data.shape)
            break
        count = count + 1

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        res = model(data)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        inference_time.append((start.elapsed_time(end))/data.shape[0]) # adding time in millisecond
        scores = softmax(res.cpu().detach().numpy(), axis=-1)
        predictions.append(scores)
        filename.append(files)

    results[str(count) + "_time"] = inference_time
    results[str(count) + "_conf"] = predictions
    results[str(count) + "_files"] = filename
    results.to_csv(csv_name)
    common_obj.get_stats(csv_name)
    print("#### PYTORCH Inference ends ####")

    return model

def run_tvm_inference(model, common_obj:CommonData):
    print("\n#### TVM Inference start ####")
    csv_name = f"TVM_enabled_{common_obj.model_name}_bch{common_obj.batch}.csv"
    inference_time = []
    predictions = []
    filename = []
    results = pd.DataFrame()

    dls = AnimalClassification(common_obj.batch)
    input_data = torch.randn((common_obj.batch,3,224,224))
    scripted_model = torch.jit.trace(model.cpu(), input_data)

    # Save scripted model
    #scripted_model.save('./saved_model/scripted_model_resnet50.pt')

    target = tvm.target.cuda(arch='sm_61')
    input_name = "data"
    shape_dict = [(input_name, (common_obj.batch,3,224,224))]

    mod, params = relay.frontend.from_pytorch(scripted_model, shape_dict)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    count = 0

    for data, labels_, files in tqdm.tqdm(dls.testloader):
        if(data.shape[0] != common_obj.batch):
            print("Ignored bath size ", data.shape)
            break
        count = count + 1
        dtype = "float32"
        module.set_input(input_name, data)
        output_shape = (data.shape[0], 3)
        start = time.time()
        module.run()
        tvm_output = module.get_output(0)
        end = time.time()
        tvm_output = tvm_output.numpy()
        inference_time.append((end-start)/data.shape[0])

        scores = softmax(tvm_output, axis=-1)
        predictions.append(scores)
        filename.append(files)


    results[str(count) + "_time"] = inference_time
    results[str(count) + "_conf"] = predictions
    results[str(count) + "_files"] = filename
    results.to_csv(csv_name)
    common_obj.get_stats(csv_name)
    print("#### TVM Inference start ####")
    return mod, module

def tvm_benchmark(model, common_obj:CommonData):
    print("\n#### TVM Benchmark start ####")
    csv_name = f"TVM_enabled_{common_obj.model_name}_bch{common_obj.batch}.csv"
    inference_time = []
    predictions = []
    filename = []
    results = pd.DataFrame()

    dls = AnimalClassification(common_obj.batch)
    input_data = torch.randn((common_obj.batch,3,224,224))
    scripted_model = torch.jit.trace(model.cpu(), input_data)

    target = tvm.target.cuda(arch='sm_61')
    input_name = "data"
    shape_dict = [(input_name, (common_obj.batch,3,224,224))]

    mod, params = relay.frontend.from_pytorch(scripted_model, shape_dict)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    count = 0

    module = common_obj.tvm_warmup(module)

    for data, labels_, files in tqdm.tqdm(dls.testloader):
        if(data.shape[0] != common_obj.batch):
            print("Ignored bath size ", data.shape)
            break
        count = count + 1

        data_tvm = tvm.nd.array(data.cpu().numpy().astype('float32'), tvm.cuda(0))
        module.set_input("data", data_tvm)
        tvm_results = module.benchmark(device=tvm.cuda(0), func_name="run", repeat=3, min_repeat_ms=500, number=3)

        inference_time.append((tvm_results.mean/data.shape[0]) * 1000)
        tvm_output = module.get_output(0).numpy()
        scores = softmax(tvm_output, axis=-1)
        predictions.append(scores)
        filename.append(files)

        #data_tvm = tvm.nd.array(data.cpu().numpy().astype('float32'), tvm.cuda(0))
        #module.set_input("data", data_tvm)
        #module.set_input(**{k:tvm.nd.array(v, tvm.cuda(0)) for k, v in params.items()})

        #module.set_input(input_name, data)

        # Evaluate
        #print("Evaluate inference time cost...")
        #https://tvm.apache.org/docs/reference/api/python/graph_executor.html#tvm.contrib.graph_executor.GraphModule.benchmark
        #print(module.benchmark(device=tvm.cuda(0), func_name="run", repeat=3, min_repeat_ms=500, number=10))
        #tvm_results = module.benchmark(device=tvm.cuda(0), func_name="run", repeat=5, min_repeat_ms=500, number=10)



        # evaluate
        #ftimer = module.module.time_evaluator("run", tvm.cuda(0), number=10, repeat=5)
        #t = ftimer(data_tvm).results
        #t = np.array(t) * 1000

        #print('{} ms'.format(t.mean()))
        #break

        #prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
        #print("mean = ", np.mean(prof_res))
        #print("std = ", np.std(prof_res))

        """ output_shape = (data.shape[0], 3)
        start = time.time()
        module.run()
        tvm_output = module.get_output(0)
        end = time.time()
        tvm_output = tvm_output.numpy()
        inference_time.append((end-start)/data.shape[0])
`
        scores = softmax(tvm_output, axis=-1)
        predictions.append(scores)
        filename.append(files)
        break"""

    results[str(count) + "_time"] = inference_time
    results[str(count) + "_conf"] = predictions
    results[str(count) + "_files"] = filename
    results.to_csv(csv_name)
    common_obj.get_stats(csv_name)

    print("#### TVM Inference start ####")
    return mod, module

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=8,
                        help="Set the batch size used in inference", type=int)

    args = parser.parse_args()

    obj = CommonData(args.batch)

    pt_model = run_pytorch_inference(obj)
    torch.save(pt_model, os.path.join(obj.model_save_path, 'resnet50.pt'))

    #mod, module = run_tvm_inference(pt_model, obj)
    mod, module = tvm_benchmark(pt_model, obj)

    from tvm_viz import visualize
    visualize(mod['main'])  # convert to png using dot -Tpng tvm_graph.dot > tvm_tvm.png