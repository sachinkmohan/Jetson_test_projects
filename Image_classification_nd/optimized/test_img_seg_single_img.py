import model_inference as mi
import engine_ops as eop
#import preprosses as pre
from os.path import isfile, join
import os
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import time
import numpy as np
import cv2 as cv


if __name__ == "__main__":

    try:

        engine_path = './models/plan/seg_model_unet_40_ep.plan'

        batch_size1 = 1
        start = time.time()
        n = 10
        for i in range(n):
            #engine_path = join(os.getcwd(), "/models/plan/ssd7_keras_1.plan")
            img_inf = cv.imread('0001TP_009060.png')
            image_resized2 = cv.resize(img_inf, (480, 320))
            frame2 = np.expand_dims(image_resized2, axis=0)
            out = mi.inference(engine_path,  frame2, batch_size1)
        end = time.time()
        print('Average runtime: %f seconds' % (float(end - start) / n))

    except BaseException as err:
        #logger.error(err)
        cv.destroyAllWindows()
        raise err
