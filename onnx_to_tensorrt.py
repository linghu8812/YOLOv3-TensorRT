from __future__ import print_function
import common
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw, ImageFont

from data_processing import PreprocessYOLO, PostprocessYOLO, load_label_categories
import calibrator

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import argparse


TRT_LOGGER = trt.Logger()


def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    np.random.seed(1)
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(all_categories))]
    text_size = 20
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", text_size)

    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        bbox_color = tuple(colors[category]) or bbox_color
        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color, width=3)
        draw.text((left, top - 20), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color, font=font)

    return image_raw


def get_engine(onnx_file_path, width=608, height=608, batch_size=1, engine_file_path="", int8mode=False,
               calib_file='yolo_calibration.cache'):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = batch_size
            if int8mode:
                # calibrator definition
                calibration_dataset_loc = "calibration_dataset/"
                calibration_cache = calib_file
                calib = calibrator.PythonEntropyCalibrator(calibration_dataset_loc, cache_file=calibration_cache,
                                                           width=width, height=height, batch_size=batch_size)
                builder.int8_mode = True
                builder.int8_calibrator = calib
            else:
                builder.fp16_mode = True
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main(width=608, height=608, batch_size=1, dataset='coco_label.txt', int8mode=False, calib_file='yolo_calibration.cache',
         onnx_file='yolov3.onnx', engine_file='yolov3.trt', image_file='dog.jpg', result_file='dog_bboxes.png'):

    """Load labels of the correspond dataset."""
    label_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset)
    all_categories = load_label_categories(label_file_path)
    classes = len(all_categories)

    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = onnx_file
    engine_file_path = engine_file
    # Download a dog image and save it to the following file path:
    input_image_path = image_file
    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (height, width)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw, image = preprocessor.process(input_image_path, batch_size)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.size

    # Output shapes expected by the post-processor
    output_shapes = [(batch_size, (classes + 5) * 3, height // 32, width // 32),
                     (batch_size, (classes + 5) * 3, height // 16, width // 16),
                     (batch_size, (classes + 5) * 3, height // 8,  width // 8)]
    # Do inference with TensorRT
    with get_engine(onnx_file_path, width, height, batch_size, engine_file_path, int8mode, calib_file) as engine, \
            engine.create_execution_context() as context:
        start = time.time()
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    end = time.time()
    print("Inference costs %.03f sec." % (end - start))
    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
                                           (59, 119), (116, 90), (156, 198), (373, 326)],
                          "obj_threshold": 0.6,                                               # Threshold for object coverage, float value between 0 and 1
                          "nms_threshold": 0.5,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    trt_outputs_1 = [np.expand_dims(trt_outputs[0][0], axis=0),
                     np.expand_dims(trt_outputs[1][0], axis=0),
                     np.expand_dims(trt_outputs[2][0], axis=0)]
    boxes, classes, scores = postprocessor.process(trt_outputs_1, (shape_orig_WH), classes)
    # Draw the bounding boxes onto the original input image and save it as a PNG file
    obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, all_categories)
    output_image_path = result_file
    obj_detected_img.save(output_image_path, 'PNG')
    print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample of YOLOv3 TensorRT.')
    parser.add_argument('--width', type=int, default=608, help='image width')
    parser.add_argument('--height', type=int, default=608, help='image height')
    parser.add_argument('--batch_size', type=int, default=1, help='image height')
    parser.add_argument('--dataset', type=str, default='coco_labels.txt', help='dataset classes names label')
    parser.add_argument('--int8', action='store_true', help='set int8 mode')
    parser.add_argument('--calib_file', type=str, default='yolo_calibration.cache', help='int8 calibration file')
    parser.add_argument('--onnx_file', type=str, default='yolov3.onnx', help='yolo onnx file')
    parser.add_argument('--engine_file', type=str, default='yolov3.trt', help='yolo tensorrt file')
    parser.add_argument('--image_file', type=str, default='dog.jpg', help='image file')
    parser.add_argument('--result_file', type=str, default='dog_bboxes.png', help='result file')
    args = parser.parse_args()
    print(args)
    main(args.width, args.height, args.batch_size, args.dataset, args.int8, args.calib_file, args.onnx_file, args.engine_file,
         args.image_file, args.result_file)
