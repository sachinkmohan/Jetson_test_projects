import engine as eng
import argparse
from onnx import ModelProto 
#import tensorrt as trt
import tensorflow.contrib.tensorrt as trt
 
 
def main():
    engine_name = 'ssd7_30_ep_op13.plan'
    #onnx_path = '/home/mohan/git/swiftnet/swiftnet/swiftnet.onnx'
    onnx_path = './ssd7_30_ep_op13.onnx'
    batch_size = 1
    
    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())
    
    a1 = model.graph.input
    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    #print('printing d0 ->', d0)
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    shape = [batch_size , d0, d1 ,d2]
    #shape = [1,1,320, 480, 3]
    engine = eng.build_engine(onnx_path, shape= shape)
    eng.save_engine(engine, engine_name) 
 
 
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--onnx_file', type=str)
    # parser.add_argument('--plan_file', type=str, default='engine.plan')
    # args=parser.parse_args()
    main()
