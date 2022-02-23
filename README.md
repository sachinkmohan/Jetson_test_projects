# Jetson Test Projects
Medium to high complexity Machine learning projects to compare the inference times between optimized and non optimized nueral networks

# 1) Image Classification using Resnet50 
# Required Libraries:
```python
python == 3.6 
tensorrt == 8.0.1.6 
pycuda == 2021.1 
numpy == 1.19.4 
opencv == 4.1.1 #(preinstalled with jetpack) .
argparse == 1.1 
onnx == 1.10.1 
tensorflow == 2.5.0 
```

#### useful links for some of the important libraries:
Tensorflow : 
1) https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html
2) https://www.youtube.com/watch?v=ynK-X5IPu1A

(Make sure to specify right jetpack version if you are following the tutorial from the link 2)

onnx (both onnxruntime and tf2onnx):
1) https://github.com/onnx/tensorflow-onnx

Tensorrt engine:
1) https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html


### Very Important Note: As both unoptimized and optimized versions are running on GPUs make sure tensorflow is using GPU 
by using the following command in python environment:

```python
import tensorflow as tf
print(len(tf.config.list_physical_devices('GPU')))
# The output should be >= 1 .
```

## Instructions to Run the tests:
### place the data into ```data/```:
dataset can be found at : https://www.kaggle.com/alessiocorrado99/animals10?select=raw-img

After placing the data : the directory should look like : data/raw-img

Download the onnx file from : https://drive.google.com/drive/u/2/folders/1yYDD2_teYR9wPLwCjV3OMMtrjWTyOBu0

### Place the onnx file in the respective paths:
onnx : optimized/models/onnx/resnet50.onnx

## Run the following commands in the project directory
### unoptimized model on gpu:
To show the info of the python program (Helps to know if all the libraries are installed correctly): 
```bash
$ bash run_keras.sh -debug 
```
Without info (Only the intended output):
```bash
$ bash run_keras.sh -no-debug 
```

### For optimized Program,

To show the info of the python program: 
```bash
$ bash run_optimized.sh -debug 
```
Without info:
```bash
$ bash run_optimized.sh -no-debug 
```

# 2) Image Segmentation
Download the onnx file and the hd5 file from the following link: https://drive.google.com/drive/folders/1NypCjzfGBJEQ6Gkj4k7n4Ipwcl9nix42?usp=sharing

Place the onnx file in ```Image_segmentation_nd/optimized/models/onnx```

Place the hd5 file in ```Image_segmentation_nd/```

Place the any one of the folders (e.g cane) from the data set downloaded for Image Classification Example in the ```Image_segmentation_nd/data/```

### Running test scripts
Tensorflow : ``` bash run_tf.sh -d no -b 1 ``` (set the flag -b to yes to show the entire output)

Tensorrt : ``` bash run_optimized.sh -d no -b 1 ``` (set the flag -b to yes to show the entire output)

The outputs will be stored in ```Image_segmentation_nd/data/out```  directory 


### Results from Keras -> Model.predict()
```
2022-01-31 21:11:54.818402: I tensorflow/core/common_runtime/placer.cc:54] Placeholder_319: (Placeholder): /job:localhost/replica:0/task:0/device:CPU:0
2022-01-31 21:11:44,700 From /home/mohan/anaconda2/envs/mltf115_3/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
2022-01-31 21:11:49,521 Loading Dataset
2022-01-31 21:11:54,570 Starting inference
2022-01-31 21:13:19,879 Total time for inference for 2112 images 85.31 seconds
2022-01-31 21:13:19,989 Keras Predicted: [('n03637318', 'lampshade', 0.8113549), ('n03724870', 'mask', 0.041457955), ('n04380533', 'table_lamp', 0.020961437), ('n02869837', 'bonnet', 0.013467569), ('n02268443', 'dragonfly', 0.007206127), ('n03935335', 'piggy_bank', 0.0068558655), ('n03476684', 'hair_slide', 0.0068135075), ('n03127747', 'crash_helmet', 0.005714845), ('n06874185', 'traffic_light', 0.005398398), ('n04259630', 'sombrero', 0.004695146)]

```

### Results from Optimized
```
(mltf115_3) mohan@autonoe:~/git/Jetson_test_projects/Image_classification_nd$ bash run_optimized.sh -debug
[01/31/2022-21:09:23] [TRT] [W] onnx2trt_utils.cpp:366: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
[01/31/2022-21:09:23] [TRT] [W] ShapedWeights.cpp:173: Weights resnet50/predictions/MatMul/ReadVariableOp:0 has been transposed with permutation of (1, 0)! If you plan on overwriting the weights with the Refitter API, the new weights must be pre-transposed.
/home/mohan/git/Jetson_test_projects/Image_classification_nd/optimized/models/plan/resnet50.plan
engine.get_binding_shape(0)   (1, 224, 224, 3)
engine.get_binding_shape(0)   (1, 1000)
resnet50/conv1_conv/Conv2D__6: 0.03072ms
Conv__435 + resnet50/conv1_relu/Relu: 0.120832ms
resnet50/pool1_pad/Pad: 0.055296ms
resnet50/pool1_pool/MaxPool: 0.044032ms
resnet50/conv2_block1_1_conv/Conv2D + resnet50/conv2_block1_1_relu/Relu: 0.028672ms
resnet50/conv2_block1_2_conv/Conv2D + resnet50/conv2_block1_2_relu/Relu: 0.069632ms
resnet50/conv2_block1_3_conv/Conv2D: 0.073728ms
resnet50/conv2_block1_0_conv/Conv2D + resnet50/conv2_block1_add/add + resnet50/conv2_block1_out/Relu: 0.099328ms
resnet50/conv2_block2_1_conv/Conv2D + resnet50/conv2_block2_1_relu/Relu: 0.058368ms
resnet50/conv2_block2_2_conv/Conv2D + resnet50/conv2_block2_2_relu/Relu: 0.069632ms
resnet50/conv2_block2_3_conv/Conv2D + resnet50/conv2_block2_add/add + resnet50/conv2_block2_out/Relu: 0.088064ms
resnet50/conv2_block3_1_conv/Conv2D + resnet50/conv2_block3_1_relu/Relu: 0.062464ms
resnet50/conv2_block3_2_conv/Conv2D + resnet50/conv2_block3_2_relu/Relu: 0.068608ms
resnet50/conv2_block3_3_conv/Conv2D + resnet50/conv2_block3_add/add + resnet50/conv2_block3_out/Relu: 0.089088ms
resnet50/conv3_block1_1_conv/Conv2D + resnet50/conv3_block1_1_relu/Relu: 0.052224ms
resnet50/conv3_block1_2_conv/Conv2D + resnet50/conv3_block1_2_relu/Relu: 0.0768ms
resnet50/conv3_block1_3_conv/Conv2D: 0.068608ms
resnet50/conv3_block1_0_conv/Conv2D + resnet50/conv3_block1_add/add + resnet50/conv3_block1_out/Relu: 0.111616ms
resnet50/conv3_block2_1_conv/Conv2D + resnet50/conv3_block2_1_relu/Relu: 0.07168ms
resnet50/conv3_block2_2_conv/Conv2D + resnet50/conv3_block2_2_relu/Relu: 0.075776ms
resnet50/conv3_block2_3_conv/Conv2D + resnet50/conv3_block2_add/add + resnet50/conv3_block2_out/Relu: 0.062464ms
resnet50/conv3_block3_1_conv/Conv2D + resnet50/conv3_block3_1_relu/Relu: 0.072704ms
resnet50/conv3_block3_2_conv/Conv2D + resnet50/conv3_block3_2_relu/Relu: 0.069632ms
resnet50/conv3_block3_3_conv/Conv2D + resnet50/conv3_block3_add/add + resnet50/conv3_block3_out/Relu: 0.062464ms
resnet50/conv3_block4_1_conv/Conv2D + resnet50/conv3_block4_1_relu/Relu: 0.07168ms
resnet50/conv3_block4_2_conv/Conv2D + resnet50/conv3_block4_2_relu/Relu: 0.068608ms
resnet50/conv3_block4_3_conv/Conv2D + resnet50/conv3_block4_add/add + resnet50/conv3_block4_out/Relu: 0.064512ms
resnet50/conv4_block1_1_conv/Conv2D + resnet50/conv4_block1_1_relu/Relu: 0.07168ms
resnet50/conv4_block1_2_conv/Conv2D + resnet50/conv4_block1_2_relu/Relu: 0.072704ms
resnet50/conv4_block1_3_conv/Conv2D: 0.069632ms
resnet50/conv4_block1_0_conv/Conv2D + resnet50/conv4_block1_add/add + resnet50/conv4_block1_out/Relu: 0.11264ms
resnet50/conv4_block2_1_conv/Conv2D + resnet50/conv4_block2_1_relu/Relu: 0.080896ms
resnet50/conv4_block2_2_conv/Conv2D + resnet50/conv4_block2_2_relu/Relu: 0.072704ms
resnet50/conv4_block2_3_conv/Conv2D + resnet50/conv4_block2_add/add + resnet50/conv4_block2_out/Relu: 0.070656ms
resnet50/conv4_block3_1_conv/Conv2D + resnet50/conv4_block3_1_relu/Relu: 0.079872ms
resnet50/conv4_block3_2_conv/Conv2D + resnet50/conv4_block3_2_relu/Relu: 0.070656ms
resnet50/conv4_block3_3_conv/Conv2D + resnet50/conv4_block3_add/add + resnet50/conv4_block3_out/Relu: 0.070656ms
resnet50/conv4_block4_1_conv/Conv2D + resnet50/conv4_block4_1_relu/Relu: 0.08192ms
resnet50/conv4_block4_2_conv/Conv2D + resnet50/conv4_block4_2_relu/Relu: 0.07168ms
resnet50/conv4_block4_3_conv/Conv2D + resnet50/conv4_block4_add/add + resnet50/conv4_block4_out/Relu: 0.070656ms
resnet50/conv4_block5_1_conv/Conv2D + resnet50/conv4_block5_1_relu/Relu: 0.08192ms
resnet50/conv4_block5_2_conv/Conv2D + resnet50/conv4_block5_2_relu/Relu: 0.07168ms
resnet50/conv4_block5_3_conv/Conv2D + resnet50/conv4_block5_add/add + resnet50/conv4_block5_out/Relu: 0.069632ms
resnet50/conv4_block6_1_conv/Conv2D + resnet50/conv4_block6_1_relu/Relu: 0.079872ms
resnet50/conv4_block6_2_conv/Conv2D + resnet50/conv4_block6_2_relu/Relu: 0.070656ms
resnet50/conv4_block6_3_conv/Conv2D + resnet50/conv4_block6_add/add + resnet50/conv4_block6_out/Relu: 0.070656ms
Reformatting CopyNode for Input Tensor 0 to resnet50/conv5_block1_1_conv/Conv2D + resnet50/conv5_block1_1_relu/Relu: 0.029696ms
resnet50/conv5_block1_1_conv/Conv2D + resnet50/conv5_block1_1_relu/Relu: 0.06656ms
Reformatting CopyNode for Input Tensor 0 to resnet50/conv5_block1_2_conv/Conv2D + resnet50/conv5_block1_2_relu/Relu: 0.013312ms
resnet50/conv5_block1_2_conv/Conv2D + resnet50/conv5_block1_2_relu/Relu: 0.118784ms
resnet50/conv5_block1_3_conv/Conv2D: 0.093184ms
resnet50/conv5_block1_0_conv/Conv2D + resnet50/conv5_block1_add/add + resnet50/conv5_block1_out/Relu: 0.22016ms
resnet50/conv5_block2_1_conv/Conv2D + resnet50/conv5_block2_1_relu/Relu: 0.094208ms
resnet50/conv5_block2_2_conv/Conv2D + resnet50/conv5_block2_2_relu/Relu: 0.116736ms
resnet50/conv5_block2_3_conv/Conv2D + resnet50/conv5_block2_add/add + resnet50/conv5_block2_out/Relu: 0.118784ms
resnet50/conv5_block3_1_conv/Conv2D + resnet50/conv5_block3_1_relu/Relu: 0.094208ms
resnet50/conv5_block3_2_conv/Conv2D + resnet50/conv5_block3_2_relu/Relu: 0.116736ms
resnet50/conv5_block3_3_conv/Conv2D + resnet50/conv5_block3_add/add + resnet50/conv5_block3_out/Relu: 0.119808ms
resnet50/avg_pool/Mean: 0.013312ms
resnet50/predictions/MatMul + resnet50/predictions/BiasAdd/ReadVariableOp:0 + (Unnamed Layer* 143) [Shuffle] + unsqueeze_node_after_resnet50/predictions/BiasAdd/ReadVariableOp:0 + (Unnamed Layer* 143) [Shuffle]_(Unnamed Layer* 143) [Shuffle]_output + resnet50/predictions/BiasAdd: 0.0768ms
copied_squeeze_after_resnet50/predictions/BiasAdd: 0.006144ms
resnet50/predictions/Softmax: 0.011264ms
Keras Predicted: [('n03637318', 'lampshade', 0.811354), ('n03724870', 'mask', 0.04145811), ('n04380533', 'table_lamp', 0.020961534), ('n02869837', 'bonnet', 0.01346767), ('n02268443', 'dragonfly', 0.007206159), ('n03935335', 'piggy_bank', 0.0068559037), ('n03476684', 'hair_slide', 0.006813543), ('n03127747', 'crash_helmet', 0.0057148677), ('n06874185', 'traffic_light', 0.0053984183), ('n04259630', 'sombrero', 0.0046951673), ('n03709823', 'mailbag', 0.0044979607), ('n03188531', 'diaper', 0.003874613), ('n02281787', 'lycaenid', 0.002587598), ('n04033901', 'quill', 0.0021336223), ('n03325584', 'feather_boa', 0.0020495702)]
2022-01-31 21:09:49,114 TRT engine loaded
2022-01-31 21:09:49,114 Loading the test dataset
2022-01-31 21:09:54,064 Starting inference
2022-01-31 21:09:54,767 Total time taken for inference for 2112images is 0.7 seconds
2022-01-31 21:09:54,876 Keras Predicted: [('n03637318', 'lampshade', 0.811354), ('n03724870', 'mask', 0.04145811), ('n04380533', 'table_lamp', 0.020961534), ('n02869837', 'bonnet', 0.01346767), ('n02268443', 'dragonfly', 0.007206159), ('n03935335', 'piggy_bank', 0.0068559037), ('n03476684', 'hair_slide', 0.006813543), ('n03127747', 'crash_helmet', 0.0057148677), ('n06874185', 'traffic_light', 0.0053984183), ('n04259630', 'sombrero', 0.0046951673), ('n03709823', 'mailbag', 0.0044979607), ('n03188531', 'diaper', 0.003874613), ('n02281787', 'lycaenid', 0.002587598), ('n04033901', 'quill', 0.0021336223), ('n03325584', 'feather_boa', 0.0020495702)]

```
