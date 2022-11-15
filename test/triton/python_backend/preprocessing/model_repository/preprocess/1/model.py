
import numpy as np
import sys
import json
import io

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

from PIL import Image
import torchvision.transforms as transforms
import os


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        self.loader = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.CenterCrop(224),
                transforms.ToTensor(), self.normalize
            ])
    def _image_preprocess(self, image_name):
        image = self.loader(image_name)
        #expand the dimension to nchw
        image = image.unsqueeze(0)
        return image

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # 1) 获取request中name为INPUT_0的tensor数据, 并转换为image类型
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            img = in_0.as_numpy()
            image = Image.open(io.BytesIO(img.tobytes()))

            # 2) 进行图片的transformer，并将结果设置为numpy类型
            img_out = self._image_preprocess(image)
            img_out = np.array(img_out)

            # 3) 构造output tesnor
            out_tensor_0 = pb_utils.Tensor("OUTPUT_0", img_out.astype(output0_dtype))

            # 4) 设置resposne
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
