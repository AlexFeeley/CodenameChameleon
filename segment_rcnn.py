## segment_rcnn
## segmentation and rcnn based on: https://towardsdatascience.com/image-segmentation-using-mask-r-cnn-8067560ed773
## uses repo https://github.com/matterport/Mask_RCNN

## seems like this would be good to use if we just filtered out objects with door or window classifications
## however I am struggling to get the rcnn stuff up and running
import sys
sys.path.append("C:/Users/nicol/OneDrive/Documents/School/SrDes/Mask_RCNN")

import cv2
import os
import numpy as np
import random
import colorsys
import argparse
import time
from mrcnn import model as modellib
from mrcnn import visualize
from samples.coco.coco import CocoConfig
import matplotlib

def main():
    print('a')

class MyConfig(CocoConfig):
    NAME = "my_coco_inference"
    # Set batch size to 1 since we'll be running inference on one image at a time.
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 0
    IMAGES_PER_GPU = 1
    
def prepare_mrcnn_model(model_path, model_name, class_names, my_config):
    classes = open(class_names).read().strip().split("\n")
    print("No. of classes", len(classes))

    hsv = [(i / len(classes), 1, 1.0) for i in range(len(classes))]
    COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(42)
    random.shuffle(COLORS)

    model = modellib.MaskRCNN(mode="inference", model_dir=model_path, config=my_config)
    model.load_weights(model_name, by_name=True)

    return COLORS, model, classes

def custom_visualize(test_image, model, colors, classes, draw_bbox, mrcnn_visualize, instance_segmentation):
    detections = model.detect([test_image], verbose=1)[0]

    if mrcnn_visualize:
        matplotlib.use('TkAgg')
        visualize.display_instances(test_image, detections['rois'], detections['masks'], detections['class_ids'], classes, detections['scores'])
        return

    if instance_segmentation:
        hsv = [(i / len(detections['rois']), 1, 1.0) for i in range(len(detections['rois']))]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(42)
        random.shuffle(colors)

    for i in range(0, detections["rois"].shape[0]):
        classID = detections["class_ids"][i]

        mask = detections["masks"][:, :, i]
        if instance_segmentation:
            color = colors[i][::-1]
        else:
            color = colors[classID][::-1]

        # To visualize the pixel-wise mask of the object
        test_image = visualize.apply_mask(test_image, mask, color, alpha=0.5)

    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

    if draw_bbox:
        for i in range(0, len(detections["scores"])):
            (startY, startX, endY, endX) = detections["rois"][i]

            classID = detections["class_ids"][i]
            label = classes[classID]
            score = detections["scores"][i]

            if instance_segmentation:
                color = [int(c) for c in np.array(colors[i]) * 255]

            else:
                color = [int(c) for c in np.array(colors[classID]) * 255]

            cv2.rectangle(test_image, (startX, startY), (endX, endY), color, 2)
            text = "{}: {:.2f}".format(label, score)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(test_image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return test_image

def perform_inference_image(image_path, model, colors, classes, draw_bbox, mrcnn_visualize, instance_segmentation, save_enable):
    test_image = cv2.imread(image_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    output = custom_visualize(test_image, model, colors, classes, draw_bbox, mrcnn_visualize, instance_segmentation)
    if not mrcnn_visualize:
        if save_enable:
            cv2.imwrite("result.png", output)
        cv2.imshow("Output", output)
        cv2.waitKey()
        cv2.destroyAllWindows()


# def load_input_image(image_path):
#     test_img = cv2.imread(image_path)
#     h, w, _ = test_img.shape

#     return test_img, h, w


# def yolov3(yolo_weights, yolo_cfg, coco_names):
#     net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
#     classes = open(coco_names).read().strip().split("\n")
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#     return net, classes, output_layers


# def perform_detection(net, img, output_layers, w, h, confidence_threshold):
#     blob = cv2.dnn.blobFromImage(img, 1 / 255., (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     layer_outputs = net.forward(output_layers)

#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in layer_outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             # Object is deemed to be detected
#             if confidence > confidence_threshold:
#                 # center_x, center_y, width, height = (detection[0:4] * np.array([w, h, w, h])).astype('int')
#                 center_x, center_y, width, height = list(map(int, detection[0:4] * [w, h, w, h]))
#                 # print(center_x, center_y, width, height)

#                 top_left_x = int(center_x - (width / 2))
#                 top_left_y = int(center_y - (height / 2))

#                 boxes.append([top_left_x, top_left_y, width, height])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     return boxes, confidences, class_ids


# def draw_boxes(boxes, confidences, class_ids, classes, img, colors, confidence_threshold, NMS_threshold):

#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, NMS_threshold)

#     FONT = cv2.FONT_HERSHEY_SIMPLEX

#     if len(indexes) > 0:
#         for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             # print(len(colors[class_ids[i]]))
#             color = colors[i]
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             # text = f"{class_ids[i]} -- {confidences[i]}"
#             text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
#             cv2.putText(img, text, (x, y - 5), FONT, 0.5, color, 2)

#     cv2.imshow("Detection", img)


# def dectection_video_file(webcam, video_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold):
#     net, classes, output_layers = yolov3(yolo_weights, yolo_cfg, coco_names)
#     colors = np.random.uniform(0, 255, size=(len(classes), 3))

#     if webcam:
#         video = cv2.VideoCapture(0)
#         time.sleep(2.0)
#     else:
#         video = cv2.VideoCapture(video_path)

#     while True:
#         ret, image = video.read()
#         h, w, _ = image.shape
#         boxes, confidences, class_ids = perform_detection(net, image, output_layers, w, h, confidence_threshold)
#         draw_boxes(boxes, confidences, class_ids, classes, image, colors, confidence_threshold, nms_threshold)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break

#     video.release()


# def detection_image_file(image_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold):
#     img, h, w = load_input_image(image_path)
#     net, classes, output_layers = yolov3(yolo_weights, yolo_cfg, coco_names)
#     colors = np.random.uniform(0, 255, size=(len(classes), 3))
#     boxes, confidences, class_ids = perform_detection(net, img, output_layers, w, h, confidence_threshold)
#     draw_boxes(boxes, confidences, class_ids, classes, img, colors, confidence_threshold, nms_threshold)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     ## Arguments to give before running
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--video', help='Path to video file', default=None)
#     ap.add_argument('--image', help='Path to the test images', default=None)
#     ap.add_argument('--camera', help='To use the live feed from web-cam', type=bool, default=False)
#     ap.add_argument('--weights', help='Path to model weights', type=str, default='yolov3.weights')
#     ap.add_argument('--configs', help='Path to model configs',type=str, default='yolov3.cfg')
#     ap.add_argument('--class_names', help='Path to class-names text file', type=str, default='coco.names')
#     ap.add_argument('--conf_thresh', help='Confidence threshold value', default=0.5)
#     ap.add_argument('--nms_thresh', help='Confidence threshold value', default=0.4)
#     args = vars(ap.parse_args())

#     image_path = args['image']
#     yolo_weights, yolo_cfg, coco_names = args['weights'], args['configs'], args['class_names']
#     confidence_threshold = args['conf_thresh']
#     nms_threshold = args['nms_thresh']

#     if image_path:
#         detection_image_file(image_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold)

#     elif args['camera'] == True or args['video']:
#         webcam = args['camera']
#         video_path = args['video']
#         dectection_video_file(webcam, video_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold)


## runs the program and displays system errors    
if __name__ == "__main__":
    try:
        main()
    except:
        print("Unexpected error:", sys.exc_info()[0])
        cv2.destroyAllWindows()