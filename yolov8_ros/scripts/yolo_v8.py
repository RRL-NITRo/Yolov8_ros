#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
import numpy as np
import os
from ultralytics import YOLO
from time import time

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes

class Yolo_Dect:
    def __init__(self):
        # Load parameters
        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov8/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        self.use_compressed = rospy.get_param('~use_compressed', False)
        conf = rospy.get_param('~conf', 0.5)
        self.visualize = rospy.get_param('~visualize', True)

        # Device configuration
        self.device = 'cpu' if rospy.get_param('/use_cpu', False) else 'cuda'

        # Load models
        self.model1 = YOLO(os.path.join(weight_path, 'new_object_detect.pt'))
        self.model2 = YOLO(os.path.join(weight_path, 'yolov8m.pt'))
        self.model1.fuse()
        self.model2.fuse()

        self.model1.to(self.device)
        self.model2.to(self.device)
        self.model1.conf = conf
        self.model2.conf = conf

        # Image subscription
        if self.use_compressed:
            self.color_sub = rospy.Subscriber(image_topic, CompressedImage, self.compressed_image_callback, queue_size=1, buff_size=52428800)
        else:
            self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=52428800)

        # Output publishers
        self.position_pub = rospy.Publisher(pub_topic, BoundingBoxes, queue_size=1)
        self.image_pub = rospy.Publisher('/yolov8/detection_image', Image, queue_size=1)

    def image_callback(self, image):
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        self.process_image()

    def compressed_image_callback(self, image):
        np_arr = np.fromstring(image.data, np.uint8)
        self.color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        self.process_image()

    def process_image(self):
        # Process image with both models
        results1 = self.model1(self.color_image, show=False, conf=0.3)
        results2 = self.model2(self.color_image, show=False, conf=0.3)

        # Combined frame for visualization
        combined_frame = self.color_image.copy()  # Use a copy of the original frame for drawing

        # Handle and publish results
        combined_frame = self.handle_results(results1, combined_frame)
        combined_frame = self.handle_results(results2, combined_frame)

        self.publish_image(combined_frame, self.color_image.shape[0], self.color_image.shape[1])

        if self.visualize:
            cv2.imshow('YOLOv8 Combined Detection', combined_frame)
            cv2.waitKey(1)

    def handle_results(self, results, frame):
        for result in results[0].boxes:
            boundingBox = BoundingBox()
            boundingBox.xmin = np.int64(result.xyxy[0][0].item())
            boundingBox.ymin = np.int64(result.xyxy[0][1].item())
            boundingBox.xmax = np.int64(result.xyxy[0][2].item())
            boundingBox.ymax = np.int64(result.xyxy[0][3].item())
            boundingBox.Class = results[0].names[result.cls.item()]
            boundingBox.probability = result.conf.item()
            
            # Draw bounding box on the frame
            cv2.rectangle(frame, (boundingBox.xmin, boundingBox.ymin), (boundingBox.xmax, boundingBox.ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{boundingBox.Class}: {boundingBox.probability:.2f}', (boundingBox.xmin, boundingBox.ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'rgb8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = width * 3
        self.image_pub.publish(image_temp)

def main():
    rospy.init_node('yolov8_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()

if __name__ == "__main__":
    main()
