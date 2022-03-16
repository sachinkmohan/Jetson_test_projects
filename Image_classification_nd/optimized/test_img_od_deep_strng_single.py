import os

import model_inference as mi
#import engine_ops as eop
#import preprosses as pre
#from os.path import isfile, join
#import os
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import time
#import logging
import numpy as np
#from ssd_encoder_decoder.ssd_output_decoder import decode_detections
import cv2 as cv

def inference_from_video():
    cap = cv.VideoCapture('/mnt/public_work/Datasets/rrlab/deep_driving/Real_bus_dataset/convert_to_video_ffmpeg/video_deep_steering.mp4')
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        resized = cv.resize(frame, (200,100))
        im3 = np.expand_dims(resized, axis=0)
        out = mi.inference(engine_path, im3, batch_size1)
        cv.imshow('im', resized)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    end = time.time()
    print('Inference time from video is ', str(np.round((end - start), 2)))
    print('done')


def inference_ten_imgs():
    start = time.time()
    rootdir = '/mnt/public_work/Datasets/rrlab/deep_driving/Real_bus_dataset/convert_to_video_ffmpeg/test-100/'
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            frame = cv.imread(os.path.join(subdir, file))
            im3 = np.expand_dims(frame, axis=0)
            out = mi.inference(engine_path, im3, batch_size1)
            print(file)
    end = time.time()
    print('Inference time for 100 images are ', str(np.round((end - start), 2)))
    print('done')

def inference_single_img():
    start = time.time()
    im2 = cv.imread('./image1_0.png')
    im3 = np.expand_dims(im2, axis=0)
    start = time.time()
    out = mi.inference(engine_path, im3, batch_size1)
    end = time.time()
    print('Inference time is ', str(np.round((end - start), 2)))
    print('done')


if __name__ == "__main__":

    try:
        engine_path = './models/plan/deep_driving.plan'
        batch_size1 = 1
        #inference_single_img()
        #inference_from_video()
        inference_ten_imgs()

    except BaseException as err:
        #logger.error(err)
        cv.destroyAllWindows()
        raise err
