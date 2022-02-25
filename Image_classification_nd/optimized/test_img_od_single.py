import model_inference as mi
import engine_ops as eop
import preprosses as pre
from os.path import isfile, join
import os
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import time
import logging
import numpy as np
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
import cv2 as cv

classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist',
           'light']  # Just so we can print class names onto the image instead of IDs
font = cv.FONT_HERSHEY_SIMPLEX

# fontScale
fontScale = 0.5

# Blue color in BGR
color = (255, 255, 0)

# Line thickness of 2 px
thickness = 1

if __name__ == "__main__":

    try:
        engine_path = './models/plan/ssd7_keras_1.plan'

        im2 = cv.imread('./1478899365487445082.jpg')
        im3 = np.expand_dims(im2, axis=0)
        start = time.time()
        batch_size1 = 1
        out = mi.inference(engine_path,  im3, batch_size1)
        y_pred = np.reshape(out, (1,-1, 18))

        #np.save('array_jetson.npy', y_pred)
        print(y_pred.shape)

        y_pred_load = np.load('array_ssd7_pc.npy')
        y_pred_decoded = decode_detections(y_pred,
                                           confidence_thresh=0.5,
                                           iou_threshold=0.45,
                                           top_k=200,
                                           normalize_coords=True,
                                           img_height=300,
                                           img_width=480)
        end = time.time()

        for box in y_pred_decoded[0]:
            xmin = box[-4]
            ymin = box[-3]
            xmax = box[-2]
            ymax = box[-1]
            #print(xmin,ymin,xmax,ymax)
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            cv.rectangle(im2, (int(xmin),int(ymin)),(int(xmax),int(ymax)), color=(0,255,0), thickness=2 )
            cv.putText(im2, label, (int(xmin), int(ymin)), font, fontScale, color, thickness)
        cv.imshow('im', im2)
        cv.waitKey(0)

        print('Inference time is ', str(np.round((end - start), 2)))
        print('done')
        # print('Keras Predicted:', decode_predictions(out, top=15)[0])
        # logger.info("Keras Predicted: " + str(decode_predictions(out, top=15)[0]))
    except BaseException as err:
        #logger.error(err)
        cv.destroyAllWindows()
        raise err
