import model_inference as mi
import engine_ops as eop
#import preprosses as pre
from os.path import isfile, join
import os
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import time
import numpy as np
import cv2 as cv

from repo_divamgupta_isk.config import IMAGE_ORDERING
from repo_divamgupta_isk.image_pre_processing import get_image_array

import pycuda.driver as cuda
import tensorrt as trt
import engine_ops as eng

def inference_from_img():
    start = time.time()
    img_inf = cv.imread('./repo_divamgupta_isk/0016E5_07965.png')
    x = get_image_array(img_inf, 640, 320,
                    ordering=IMAGE_ORDERING)

    y = np.array([x])
    #out = mi.inference(engine_path, x, batch_size)
    start = time.time()
    #out = mi.inference_seg(engine_path,  pre_pro, batch_size1)
    np.copyto(h_input, np.asarray(x).ravel())
    context = engine.create_execution_context()

    cuda.memcpy_htod_async(d_input, h_input, stream)
    #context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    output = h_output.reshape(np.concatenate(([1], engine.get_binding_shape(1))))

    end = time.time()


    out2 = np.reshape(output,(1,-1,50))[0]
    pr = out2.reshape((160, 320, 50)).argmax(axis=2)

    cv.imwrite('./repo_divamgupta_isk/out2.jpg',pr)
    pred_image = 255 * pr.squeeze()
    u8 = pred_image.astype(np.uint8)
    im_color = cv.applyColorMap(u8, cv.COLORMAP_TURBO)
    cv.imwrite('./repo_divamgupta_isk/color_map.jpg',im_color)
    #end = time.time()
    print('Average runtime: %f seconds' % (float(end - start)))

def inference_from_video():
    cap = cv.VideoCapture('/home/mohan/git/backups/drive_1_min_more_cars.mp4')
    prev_frame_time = 0 #calculating prev frame time, ref: https://www.geeksforgeeks.org/python-displaying-real-time-fps-at-which-webcam-video-file-is-processed-using-opencv/
    new_frame_time = 0 # calculating new frame time
    while cap.isOpened():
        ret, frame = cap.read()
        x = get_image_array(frame, 640, 320,
                            ordering=IMAGE_ORDERING)

        #y = np.array([x])
        #out = mi.inference(engine_path, x, batch_size)
        start = time.time()
        #out = mi.inference_seg(engine_path,  pre_pro, batch_size1)
        np.copyto(h_input, np.asarray(x).ravel())
        context = engine.create_execution_context()

        cuda.memcpy_htod_async(d_input, h_input, stream)
        #context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        output = h_output.reshape(np.concatenate(([1], engine.get_binding_shape(1))))

        end = time.time()


        out2 = np.reshape(output,(1,-1,50))[0]
        pr = out2.reshape((160, 320, 50)).argmax(axis=2)

        pred_image = 255 * pr.squeeze()
        u8 = pred_image.astype(np.uint8)
        im_color = cv.applyColorMap(u8, cv.COLORMAP_TURBO)
        cv.imshow('im', im_color)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


if __name__ == "__main__":

    batch_size = 1
    engine_path = './repo_divamgupta_isk/vgg_unet_im_seg_base_ep20_8034.plan'
    engine = eng.load_engine(engine_path)
    a = engine.get_binding_shape(0)
    b = engine.get_binding_shape(1)
    h_input = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
    dummy = trt.volume(engine.get_binding_shape(1))
    h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    bindings = []

    try:
        #inference_from_img()
        inference_from_video()

    except BaseException as err:
        #logger.error(err)
        cv.destroyAllWindows()
        raise err
