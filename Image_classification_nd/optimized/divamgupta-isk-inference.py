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

if __name__ == "__main__":

    try:

        engine_path = './repo_divamgupta_isk/vgg_unet_im_seg_base_ep20.plan'

        batch_size1 = 1
        start = time.time()
        img_inf = cv.imread('./repo_divamgupta_isk/0016E5_07965.png')
        x = get_image_array(img_inf, 640, 320,
                        ordering=IMAGE_ORDERING)


        out = mi.inference(engine_path, np.array([x]), batch_size1)
        out2 = np.reshape(out,(1,-1,50))[0]
        pr = out2.reshape((160, 320, 50)).argmax(axis=2)

        cv.imwrite('./repo_divamgupta_isk/out2.jpg',pr)
        pred_image = 255 * pr.squeeze()
        u8 = pred_image.astype(np.uint8)
        im_color = cv.applyColorMap(u8, cv.COLORMAP_AUTUMN)
        cv.imwrite('./repo_divamgupta_isk/out1.jpg',im_color)
        end = time.time()
        print('Average runtime: %f seconds' % (float(end - start)))

    except BaseException as err:
        #logger.error(err)
        cv.destroyAllWindows()
        raise err
