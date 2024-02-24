# ML-GSC
---------------------
## Overview
The ML-GSC repository focuses on addressing the feature requirements of the Cyclonify (scan) feature in the Circlo app. To achieve this, we harness the power of the TensorFlow Object Detection API to construct a machine learning model tailored to our specific needs, you can see the documentation of TensorFlow Object Detection API through this link [**TensorFlow Object Detection API**](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md). Our model of choice is the SSD MobileNetV2 320x320 FPNLite, renowned for its efficient computational capabilities coupled with high accuracy.

 ## Architecture
![architecture](https://raw.githubusercontent.com/aldebarankwsuperrr/struktur_data/main/mobilenet.jpg)

## SSD MobileNetV2 320x320 FPNLite Model 
1. **Single Shot Multibox Detector (SSD):**
   - **Objective:** SSD is an object detection algorithm designed to identify objects in a single pass through the network. It achieves this by predicting bounding boxes and associated class scores for multiple default boxes at each spatial location in the feature maps.
   - **Multi-scale feature maps:** SSD employs feature maps of different resolutions to detect objects at various scales, capturing both small and large objects in the input image.

2. **MobileNetV2 Backbone:**
   - **Feature extractor:** MobileNetV2 serves as the backbone of the SSD architecture, extracting hierarchical features from the input image.
   - **Depthwise separable convolutions:** MobileNetV2 utilizes depthwise separable convolutions, a type of convolutional operation that reduces computational cost while preserving the expressive power of the network. This is particularly beneficial for efficient execution on mobile and edge devices.

3. **Feature Pyramid Network Lite (FPNLite):**
   - **Objective:** FPNLite enhances the feature hierarchy by introducing lateral connections to build a feature pyramid. This aids in capturing objects of different sizes by combining information from various levels of abstraction.
   - **Lite version:** FPNLite is a streamlined version of the original Feature Pyramid Network (FPN), designed to strike a balance between accuracy and computational efficiency. It ensures effective model execution on devices with limited resources.

4. **320x320 Input Size:**
   - **Input resolution:** The model is optimized for processing images with a resolution of 320x320 pixels. The choice of input size considers the need for real-time or near-real-time processing, striking a balance between accuracy and computational efficiency.

## Source
- https://vidishmehta204.medium.com/object-detection-using-ssd-mobilenet-v2-7ff3543d738d
