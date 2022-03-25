import numpy as np
import tensorrt as trt
import engine_ops as eng
import inference as inf
import pycuda.autoinit

import pycuda.driver as cuda


def initialize(engine_path, data_set, batch_size):
    engine = eng.load_engine(engine_path)
    h_input, d_input, h_output, d_output, stream = inf .allocate_buffers(engine, batch_size, trt.float32)
    return engine, h_input, d_input, h_output, d_output, stream

def inference(engine_path, data_set, batch_size):
    engine, h_input, d_input, h_output, d_output, stream = initialize(engine_path, data_set, batch_size)
    out = inf.do_inference(engine, data_set, h_input, d_input, h_output, d_output, stream, batch_size)
    return out

def inference_seg(engine_path, data_set, batch_size):
    #engine, h_inputs, cuda_inputs, h_outputs, cuda_outputs, stream = initialize_seg(engine_path, data_set, batch_size)
    engine, h_input, d_input, h_output, d_output, stream = initialize(engine_path, data_set, batch_size)
    bindings = []
    np.copyto(h_input, data_set.ravel())
    stream = cuda.Stream()
    context = engine.create_execution_context()

    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    output = h_output.reshape(np.concatenate(([1], engine.get_binding_shape(1))))
    return output