
import sys
sys.path.insert(1, '/home/artpark/UGV/VO/Visual_Odometry/feature_extractors/r2d2/')
import glob
import time
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2
import torch
# from PIL import Image
# from tools.dataloader import norm_RGB
import TRT.common 

#For debug
def breakpoint():
    inp = input("Waiting for input...")


ONNX_FILE_PATH = "./TRT/ONNX/r2d2_WASF_N16_kitti.onnx"
# logger to capture errors, warnings, and other information during the build and inference phases

class ModelData(object):
    # MODEL_PATH = "ResNet50.onnx"
    # INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

class trt_infer:
    def __init__(self):
        
        print("Inside constructor")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        # self.cfx = cuda.Device(0).make_context()
        #Read engine file
        with open("/home/artpark/UGV/VO/Visual_Odometry/TRT/engines/r2d2_WASF_N16_kitti_fp16.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
	        self.engine = runtime.deserialize_cuda_engine(f.read())
        #Prepare buffer
        self.inputs, self.outputs, self.bindings, self.stream = TRT.common.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()
        # if self.cfx:
            # print("Inside self.cfx")
            # self.cfx.pop()
    
    def infer_v1(self,frame):
        with self.engine.create_execution_context() as context:
        # Set input shape based on image dimensions for inference
            context.set_binding_shape(self.engine.get_binding_index("input"), (1, 3, frame.shape[0], frame.shape[1]))
            # Allocate host and device buffers
            bindings = []
            for binding in self.engine:
                # print("First breakpoint inside for loop")
                binding_idx = self.engine.get_binding_index(binding)
                print("binding_idx:",binding_idx)
                size = trt.volume(context.get_binding_shape(binding_idx))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                if self.engine.binding_is_input(binding):
                    input_buffer = np.ascontiguousarray(frame)
                    input_memory = cuda.mem_alloc(frame.nbytes)
                    bindings.append(int(input_memory))
                else:
                    output_buffer = cuda.pagelocked_empty(size, dtype)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    bindings.append(int(output_memory))

            stream = cuda.Stream()
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            
            # Transfer prediction output from the GPU.
            cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
            # Synchronize the stream
            stream.synchronize()
            print("output_buffer.shape",output_buffer.shape)
        # return img

    def infer(self,frame):
        # if self.cfx:
            # print("Inside self.cfx infer")
            # self.cfx.push()
        #restore 
        # stream = self.stream
        # context = self.context
        # engine = self.engine

        # inputs = self.inputs
        # outputs = self.outputs
        # bindings = self.bindings
        # stream = self.stream
    

        frame = frame.numpy()
        frame = frame.astype(trt.nptype(ModelData.DTYPE)).ravel()
        # inputs[0].host = frame
        np.copyto(self.inputs[0].host, frame)
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        res = TRT.common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        # res = {k:[r[k] for r in res if k in r] for k in {k for r in res for k in r}}
        
        # if self.cfx:
            # print("Finished self.cfx infer")
            # self.cfx.pop()
        # print("Finished infer")

        return res




def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    TRT_LOGGER= trt.Logger()
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
    # image = cv2.resize(frame, (640,480), interpolation = cv2.INTER_LANCZOS4)
    img_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_pil)
    img_cpu = img_pil
    img = norm_RGB(img_cpu)[None]
    img = img.numpy()
    return img

def r2d2_trt_inference(frame):
    start_time = time.time()
    output = trt_infer_obj.infer(frame)
    # descriptors = torch.from_numpy(np.reshape(output[0],(1,1,128,frame.shape[2],frame.shape[3]))).cuda()
    # reliability = torch.from_numpy(np.reshape(output[1],(1,1,1,frame.shape[2],frame.shape[3]))).cuda()
    # repeatability = torch.from_numpy(np.reshape(output[2],(1,1,1,frame.shape[2],frame.shape[3]))).cuda()

    descriptors = torch.from_numpy(np.reshape(output[0],(1,1,128,frame.shape[2],frame.shape[3])))
    reliability = torch.from_numpy(np.reshape(output[1],(1,1,1,frame.shape[2],frame.shape[3])))
    repeatability = torch.from_numpy(np.reshape(output[2],(1,1,1,frame.shape[2],frame.shape[3])))
    res = {'descriptors':descriptors, 'reliability':reliability, 'repeatability':repeatability}
    print("Inference time:",time.time() - start_time)

    # print("res:",res)
    return res

#Global variables
trt_infer_obj = trt_infer()

def main():
    # initialize TensorRT engine and parse ONNX model
    # engine, context = build_engine(ONNX_FILE_PATH)
    #Save to file
    # with open("r2d2_WASF_N16_kitti.engine","wb") as f:
        # f.write(engine.serialize())
    
    trt_infer_obj = trt_infer()
    for file_path in sorted(glob.glob("/media/usb/VO/data/03/image_2/*.png")):
    #    print("Inside image read")
        frame = cv2.imread(file_path)
        frame = np.expand_dims(frame,0).reshape(1,3,375,1242)
        # frame = prep_img(frame) #1x3x480x640
        print("frame.shape",frame.shape)
        start_time = time.time()
        output = trt_infer_obj.infer(frame)
        print("type(output[0]):",type(output[0]))
        # descriptors = torch.from_numpy(np.reshape(output[0],(1,128,frame.shape[2],frame.shape[3]))).cuda()
        # reliability = torch.from_numpy(np.reshape(output[1],(1,1,frame.shape[2],frame.shape[3]))).cuda()
        # repeatability = torch.from_numpy(np.reshape(output[2],(1,1,frame.shape[2],frame.shape[3]))).cuda()
        # res = {'descriptors':descriptors, 'reliability':reliability, 'repeatability':repeatability}
        print("Inference time:",time.time() - start_time)
    #    # print("res:",res)
    #    # breakpoint()

    # with open("/home/volta-2/VO/engines/r2d2.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
	#     engine = runtime.deserialize_cuda_engine(f.read())
    # context = engine.create_execution_context()

    # inputs, outputs, bindings, stream = TRT.common.allocate_buffers(engine)
    # with engine.create_execution_context() as context:
            # For more information on performing inference, refer to the introductory samples.
            # The TRT.common.do_inference function will return a list of outputs - we only have one in this case.
    # for file_path in glob.glob("/home/volta-2/VO/data/mar8seq/00/*.png"):
    #     try:
    #         frame = cv2.imread(file_path)
    #         frame = prep_img(frame)
    #         inputs[0].host = frame
    #         start_time = time.time()
    #         output = TRT.common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    #         print(output)
    #     except KeyboardInterrupt:
    #         cfx.pop()


if __name__ == '__main__':
    main()
