## Object Detection 

![](http://i.imgur.com/9xApnEN.png)

- Region 기반 딥러닝 : R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN

- Grid 기반 딥러닝 : YOLO, DetectNet
    -  YOLO는 이미지 내의 bounding box와 class probability를 single regression problem으로 간주하여, 이미지를 한 번 보는 것으로 오브젝트의 종류와 위치를 추측합니다
    - 아래와 같이 single convolutional network를 통해 multiple bounding box에 대한 class probability를 계산하는 방식을 취합니다.


|년도|알고리즘|링크|입력|출력|특징|
|-|-|-|-|-|-|
|2014|R-CNN|[논문](https://arxiv.org/abs/1311.2524)|Image|Bounding boxes + labels for each object in the image.|AlexNet, 'Selective Search'사용 |
|2015|Fast R-CNN|[논문](https://arxiv.org/abs/1504.08083)|Images with region proposals.|Object classifications |Speeding up and Simplifying R-CNN, RoI Pooling|
|2016|Faster R-CNN|[논문](https://arxiv.org/abs/1506.01497)| CNN Feature Map.|A bounding box per anchor|MS, Region Proposal|
||YOLO|||||
||SSD||||Faster R-CNN + YOLO|

> 2017년 [Mask R-CNN](https://arxiv.org/abs/1703.06870)이 발표 되었지만 Segmentation 분야여서 포함 안함 

# RCNN
Approaches using RCNN-trained models in multi-stage pipelines (first detecting object boundaries and then performing identification) 
 - rather slow and not suited for real time processing. 

The drawback of this approach is mainly its __speed__, both during the training and during the actual testing while object detection was performed. 
    - eg. VGG16, the training process for a standard RCNN takes 2.5 GPU-days for the 5k images and requires hundreds of GB of storage. Detecting objects at test-time takes 47s/image using a GPU. This is mainly caused by performing a forward pass on the convolutional network for each object proposal, without sharing the computation.

# Fast R-CNN

Fast R-CNN improved RCNN by introducing a single-stage training algorithm which classifies objects and their spatial locations in a single processing stage. The improvements introduced in Fast R-CNN are:
- Higher detection quality
- Training in a single stage using multi-task loss
- Training can update all network layers
- No disk storage is required for feature caching

# Faster R-CNN

Faster R-CNN introduces a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, enabling nearly cost-free region proposals. The RPN component of this solution tells the unified network where to look. For the same VGG-16 model, Faster R-CNN has a frame rate of 5 fps on a GPU while achieving state-of-the-art object detection accuracy. The RPN is a kind of a fully convolutional network and can be trained end-to-end specifically for the task of generating detection proposals and is designed to efficiently predict region proposals with a wide range of scales and aspect ratios. [[Code]](https://github.com/softberries/keras-frcnn)


> 출처 : [Counting Objects with Faster R-CNN](https://softwaremill.com/counting-objects-with-faster-rcnn/)

# YOLO

- Super fast detector (21~155 fps)

- Finding objects at each grid __in parallel__

- 성능 : Fast R-CNN < YOLO < Faster R-CNN 




# SSD 

- Faster R-CNN + YOLO

- Multi-scale feature map detection 
    - Detect small objects on lower level, large objects on higher level
    
- End-to-End training/testing 




# 참고 자료 

- [Selective Search for Object Recognition](https://www.koen.me/research/pub/uijlings-ijcv2013-draft.pdf)

- Recognition using Regions

- [Regression Methods for Localization](https://bfeba431-a-62cb3a1a-s-sites.googlegroups.com/site/deeplearningcvpr2014/RegressionMethodsforLocalization.pdf?attachauth=ANoY7cpf41j03XW6YUpHg5L5_LgNhz6C05lpU58CkgQixIXesT0WOK6HU3CVi5x8t83aWcvYkvrUIpZ80rXYI8Hnlfk-wFdcay_DWW4c9ww5KXDADhcyhMiCDOv3AnNkhmuQDLFWCxyjY--VParh1WCIVUIOvtj4NW_UPc2zz0I_b9ovWkK-_qEio3oAY29Z6cyzK4Co60biKGRrc_3WfXxJgdq0Zq7pPnopAAHdEFpU9bv360H-EeW88n-h--8fyCQhJsG7-Pm-&attredirects=0): ppt

- J. Long, “Fully convolutional networks for semantic segmentation,” in IEEE CVPR 2015. :[Github](https://github.com/shelhamer/fcn.berkeleyvision.org), [한글분석](http://www.whydsp.org/317), [Arxiv](https://arxiv.org/abs/1605.06211), [ppt](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-pixels.pdf)

- K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition,
CVPR 2016. (R-CNNs are based on ResNets)


--- 

# 성능향상 기법 

- Hard negative mining을 이용한 2~3% 성능 향상 방법들에 대한 서베이 논문 
    - [Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/abs/1604.03540)