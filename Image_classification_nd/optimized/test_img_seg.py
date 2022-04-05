import model_inference as mi
import engine_ops as eop
#import preprosses as pre
from os.path import isfile, join
import os
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import time
#import logging
import numpy as np
#from ssd_encoder_decoder.ssd_output_decoder import decode_detections
import cv2 as cv
#from PIL import Image

import tensorrt as trt
import engine_ops as eng
import inference as inf
import pycuda.driver as cuda

import tensorflow as tf
import segmentation_models as sm

if __name__ == "__main__":

    '''
    logging.basicConfig(filename="optimized/logs_trt.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

    #Creating an object
    logger=logging.getLogger()

    #Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    #Test messages
    ##save engine
    '''
    
    BACKBONE = 'efficientnetb3' 
    preprocess_input = sm.get_preprocessing(BACKBONE) # added for preprocessing 
    try:
        # engine_path = join(os.getcwd(),"optimized/models/plan")
        #img1 = cv.imread('../data/data1/berlin_001.png')
        cap = cv.VideoCapture('drive.mp4')

        prev_frame_time = 0
        new_frame_time = 0
        '''
        engine_path = join(os.getcwd(), "Image_classification_nd/optimized/models/plan")
        if (len(os.listdir(engine_path)) == 0 or os.listdir(engine_path) != 'ssd7_keras_1.plan'):
            onnx_path = join(os.getcwd(), 'Image_classification_nd/optimized/models/onnx/ssd7_keras_1.onnx')
            engine_path = join(os.getcwd(), "Image_classification_nd/optimized/models/plan/ssd7_keras_1.plan")
            eop.save_engine(engine_path, onnx_path)
        else:
            engine_path = join(os.getcwd(), "optimized/models/plan/ssd7_keras_1.plan")
        '''
        #logger.debug("TRT engine loaded")
        #logger.debug("Loading the test dataset")
        # os.path.dirname(os.getcwd()) to go one level up
        # data_path = join(os.getcwd(),"Image_classification_nd/data/raw-img-2")
        # data_set = pre.return_dataset(data_path)
        # data_set = data_set.reshape(data_set.shape[1],data_set.shape[2], data_set.shape[3], data_set.shape[4])
        # data_set = data_set.reshape(data_set.shape[1],300, 480, data_set.shape[4])

        #engine_path = './models/plan/seg_model_unet_40_ep_op13.plan'
        engine_path = './seg_model_unet_40_ep_op13_v803.plan'

        font = cv.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame

        def initialize(engine_path, data_set, batch_size):
            engine = eng.load_engine(engine_path)
            h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, batch_size, trt.float32)
            return engine, h_input, d_input, h_output, d_output, stream

        batch_size=1
        engine = eng.load_engine(engine_path)
        h_input = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
        h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_input.nbytes)
        stream = cuda.Stream()
        bindings = []

        while cap.isOpened():
            #engine_path = join(os.getcwd(), "/models/plan/ssd7_keras_1.plan")

            new_frame_time = time.time()


            # putting the FPS count on the frame
            ret, frame = cap.read()
            resized = cv.resize(frame, (480, 320))
            #im3 = np.expand_dims(resized, axis=0)
            img_inf = preprocess_input(resized)
            #pre_pro = (2.0 / 255.0) * resized.transpose((2, 0, 1)) - 1.0  # Converting HWC -> CHW

            # Ref 1 -> https://elinux.org/Jetson/L4T/TRT_Customized_Example#OpenCV_with_PLAN_model
            # Ref 2 -> https://github.com/NVIDIA/object-detection-tensorrt-example/blob/master/SSD_Model/utils/inference.py

            start = time.time()
            batch_size1 = 1
            #out = mi.inference_seg(engine_path,  pre_pro, batch_size1)
            np.copyto(h_input, np.asarray(img_inf).ravel())
            context = engine.create_execution_context()

            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            output = h_output.reshape(np.concatenate(([1], engine.get_binding_shape(1))))

            end = time.time()

            output_image = output.reshape((320, 480, -1))
            #pred = 255*np.argmax(output_image, -1)
            #pred = np.uint8(pred)
            pred_image = 255*output_image.squeeze()
            u8 = pred_image.astype(np.uint8)
            im_color = cv.applyColorMap(u8, cv.COLORMAP_AUTUMN)

            # Calculating the fps

            # fps will be number of frame processed in given time frame
            # since their will be most of time error of 0.001 second
            # we will be subtracting it to get more accurate result
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # converting the fps into integer
            fps = int(fps)

            # converting the fps to string so that we can display it on frame
            # by using putText function
            fps = str(fps)


            cv.putText(resized, fps, (7, 70), font, 3, (100, 255, 0), 3, cv.LINE_AA)
            cv.imshow('input_image', resized)
            cv.imshow('output_image', im_color)


            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            print('Inference time swiftnet ', str(np.round((end - start), 2)))

            # print('Keras Predicted:', decode_predictions(out, top=15)[0])
            # logger.info("Keras Predicted: " + str(decode_predictions(out, top=15)[0]))
    except BaseException as err:
        #logger.error(err)
        cv.destroyAllWindows()
        raise err
