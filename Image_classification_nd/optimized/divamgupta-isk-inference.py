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

        engine_path = './repo-divamgupta-isk/vgg_unet_im_seg_base_ep20.plan'

        batch_size1 = 1
        start = time.time()
        img_inf = cv.imread('./repo-divamgupta-isk/converted_img.png')
        
        out = mi.inference(engine_path, np.array([img_inf]), batch_size1)
        out2 = np.reshape(out,(1,-1,50))
        pr = out2.reshape((160, 320, 50)).argmax(axis=2)

        cv.imwrite('./repo-divamgupta-isk/out2.jpg',pr)
        pred_image = 255 * pr.squeeze()
        u8 = pred_image.astype(np.uint8)
        im_color = cv.applyColorMap(u8, cv.COLORMAP_AUTUMN)
        cv.imwrite('./repo-divamgupta-isk/out1.jpg',im_color)
        end = time.time()
        print('Average runtime: %f seconds' % (float(end - start)))

    except BaseException as err:
        #logger.error(err)
        cv.destroyAllWindows()
        raise err
