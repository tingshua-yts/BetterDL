# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os, sys
import numpy as np
import json
import tritongrpcclient
import argparse
import time


def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.

    """
    return np.fromfile(img_path, dtype='uint8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        required=False,
                        default="ensemble_python_resnet50",
                        help="Model name")
    parser.add_argument("--image",
                        type=str,
                        required=True,
                        help="Path to the image")
    parser.add_argument("--url",
                        type=str,
                        required=False,
                        default="localhost:8001",
                        help="Inference server URL. Default is localhost:8001.")
    parser.add_argument('-v',
                        "--verbose",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument(
        "--label_file",
        type=str,
        default="./model_repo/resnet50/labels.txt",
        help="Path to the file with text representation of available labels")
    args = parser.parse_args()

    try:
        triton_client = tritongrpcclient.InferenceServerClient(
            url=args.url, verbose=args.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    with open(args.label_file) as f:
        labels_dict = {idx: line.strip() for idx, line in enumerate(f)}



    input_name = "INPUT"
    output_name = "OUTPUT"

    # 1）加载本地image，构建为一个一维的数组，转化为numpy
    image_data = load_image(args.image)
    image_data = np.expand_dims(image_data, axis=0)

    # 2) 设置input
    inputs = []
    inputs.append(tritongrpcclient.InferInput(input_name, image_data.shape, "UINT8"))
    inputs[0].set_data_from_numpy(image_data)


    # 3) 设置output
    outputs = []
    # 没有添加class count，返回的是float类型的tensor
    """
    [[ ...
      -2.60193777e+00 -1.27933168e+00 -1.12860894e+00 -4.24509972e-01
       2.34700441e+00  5.82997799e-01  1.94415107e-01  3.19028640e+00]]
    """
    outputs.append(tritongrpcclient.InferRequestedOutput(output_name))

    # 添加上class_count后会在output0_data中显示class的名称，如下：
    """
            [[b'17.309477:504:COFFEE MUG'
            b'14.282034:968:CUP'
            b'11.696501:967:ESPRESSO']]
    """
    #outputs.append(tritongrpcclient.InferRequestedOutput(output_name, class_count=3))


    # 4) 发起请求
    start_time = time.time()
    results = triton_client.infer(model_name=args.model_name, inputs=inputs,  outputs=outputs)
    latency = time.time() - start_time
    # 5) 结果处理：转化为numpy 类型，计算max，转化label
    output0_data = results.as_numpy(output_name)
    # 获取outputs结果中，value最大值的下标
    maxs = np.argmax(output0_data, axis=1)
    #print(output0_data)
    print(maxs)
    """
        [504]
    """
    # 转化为对应的标签
    #print("Result is class: {}".format(labels_dict[maxs[0]]))
    print(f"{latency * 1000}ms class: {labels_dict[maxs[0]]}")
    """
        Result is class: COFFEE MUG
    """
