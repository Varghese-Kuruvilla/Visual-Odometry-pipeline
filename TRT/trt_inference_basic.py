
import sys
sys.path.insert(1, 'r2d2')
import glob
import time
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2
import torch
from PIL import Image
from tools.dataloader import norm_RGB
import common 
import logging
logger = logging.getLogger('module_trtinfer')
logger.setLevel(logging.INFO)

#For debug
def breakpoint():
    inp = input("Waiting for input...")


ONNX_FILE_PATH = "/home/volta-2/VO/Visual_Odometry_code/r2d2.onnx"
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

# with open("r2d2.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
# 	engine = runtime.deserialize_cuda_engine(f.read())
# context = engine.create_execution_context()
# cfx = cuda.Device(0).make_context()

# inputs, outputs, bindings, stream = common.allocate_buffers(engine)

def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    config.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        res = parser.parse(model.read())
        print("result:",res)
    print('Completed parsing of ONNX file')

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)

    # print("network.num_layers",network.num_layers)
    # last_layer = network.get_layer(network.num_layers)
    # network.mark_output(last_layer.get_output(0))

    context = engine.create_execution_context()
    print("Completed creating Engine")
    return engine, context


def prep_img(frame):
    image = cv2.resize(frame, (640,480), interpolation = cv2.INTER_LANCZOS4)
    img_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_pil)
    img_cpu = img_pil
    img = norm_RGB(img_cpu)[None]
    img = img.numpy()
    return img

def main():
    # initialize TensorRT engine and parse ONNX model
    # engine, context = build_engine(ONNX_FILE_PATH)
    #Save to file
    # with open("r2d2.engine","wb") as f:
    #     f.write(engine.serialize()
    
    
    with open("/workspace/VO/engines/r2d2_fp32.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
	    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    with engine.create_execution_context() as context:
            print("Inside context")
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            for file_path in sorted(glob.glob("/workspace/VO/data/mar8seq/00/left*")):
                print("Inside image read")
                frame = cv2.imread(file_path)
                frame = prep_img(frame) #1x3x480x640
                inputs[0].host = frame
                start_time = time.time()
                output = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                
                descriptors = torch.from_numpy(np.reshape(output[0],(1,128,frame.shape[2],frame.shape[3]))).cuda()
                reliability = torch.from_numpy(np.reshape(output[1],(1,1,frame.shape[2],frame.shape[3]))).cuda()
                repeatability = torch.from_numpy(np.reshape(output[2],(1,1,frame.shape[2],frame.shape[3]))).cuda()
                res = {'descriptors':descriptors, 'reliability':reliability, 'repeatability':repeatability}
                # print("res",res)
                print("Inference:",time.time()-start_time)
                # breakpoint()


if __name__ == '__main__':
    main()
