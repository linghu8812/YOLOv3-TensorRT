python3 yolov3_to_onnx.py --cfg_file yolov3.cfg --weights_file yolov3.weights --output_file yolov3.onnx  # generate onnx model

python3 onnx_to_tensorrt.py --onnx_file yolov3.onnx --engine_file yolov3.trt  # generate TensorRT model
