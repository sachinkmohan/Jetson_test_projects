import model_inference as mi
import engine_ops as eop
import preprosses as pre
from os.path import isfile, join
import os
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
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
    try:
        # engine_path = join(os.getcwd(),"optimized/models/plan")
        #img1 = cv.imread('../data/data1/two.jpg')
        cap = cv.VideoCapture('drive.mp4')

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

        engine_path = './models/plan/ssd7_30_ep_op13_v803.plan'


        while cap.isOpened():
            #engine_path = join(os.getcwd(), "/models/plan/ssd7_keras_1.plan")

            ret, frame = cap.read()
            resized = cv.resize(frame, (480, 300))
            im3 = np.expand_dims(resized, axis=0)
            

            #print('IM3 shape', im3.shape)
            #data_set = data_set.reshape(1, 300, 480, 3)
            #logger.debug("Starting inference")
            start = time.time()
            batch_size1 = 1
            out = mi.inference_seg(engine_path,  im3, batch_size1)
            y_pred = np.reshape(out, (1,-1, 18))
            print(y_pred.shape)
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
                cv.rectangle(resized, (int(xmin),int(ymin)),(int(xmax),int(ymax)), color=(0,255,0), thickness=2 )
                cv.putText(resized, label, (int(xmin), int(ymin)), font, fontScale, color, thickness)
            cv.imshow('im', resized)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            #Wait until q is pressed to exit from the video
            #logger.debug("Total time taken for inference for " + str(data_set.shape[0]) + "images is " + str(
            #    np.round((end - start), 2)) + " seconds")
            print('Inference time is ', str(np.round((end - start), 2)))
            print('done')
            # print('Keras Predicted:', decode_predictions(out, top=15)[0])
            # logger.info("Keras Predicted: " + str(decode_predictions(out, top=15)[0]))
    except BaseException as err:
        #logger.error(err)
        cv.destroyAllWindows()
        raise err
