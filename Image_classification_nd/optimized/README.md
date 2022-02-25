

#### Instructions

- Find the video link [here](https://drive.google.com/file/d/1zWBpfBLdnk7_wAtkwJfZ--SJtzQ1_Lan/view)
- First generate an ONNX file. A sample on how to generate one is given in `loadmodel_seg.py` file
	- I generated the ONNX file in Tensorflow > 2.0 environment. The latest opset which worked on the jetson board was `opset=13`. Once the ONNX file is generated, you can run your model in TF < 2.0 version as well. The ONNX file doesn't generate the input dimensions if you use TF < 2.0 and the engine file will fail to generate. 
- Once the ONNX file is generated. Generate the engine file using `buildEngine.py`
- Once the `.plan`(engine file) is generated, run the sample `test_img_seg.py` or `test_img.py` files
