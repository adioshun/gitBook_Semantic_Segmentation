![](http://i.imgur.com/ZU6afyt.png)


|년도|알고리즘|링크|입력|출력|특징|
|-|-|-|-|-|-|
|2014|R-CNN|[논문](https://arxiv.org/abs/1311.2524)|Image|Bounding boxes + labels for each object in the image.|AlexNet, 'Selective Search'사용 |
|2015|Fast R-CNN|[논문](https://arxiv.org/abs/1504.08083)|Images with region proposals.|Object classifications |Speeding up and Simplifying R-CNN, RoI Pooling|
|2016|Faster R-CNN|[논문](https://arxiv.org/abs/1506.01497),[한글](https://curt-park.github.io/2017-03-17/faster-rcnn/)| CNN Feature Map.|A bounding box per anchor|MS, Region Proposal|
|2017|Mask R-CNN|[논문](https://arxiv.org/abs/1703.06870)|CNN Feature Map.|Matrix with 1s on all locations|Facebook, pixel level|

> 출처 : [A Brief History of CNNs in Image Segmentation: From R-CNN to Mask R-CNN](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)



|R-CNN Series|YOLO/SSD|
|-|-|
|2-step|1-Step Process|
|process of region proposal and object recognition|Do region proposal and classification at same time|

> 출처 : [Recent Progress on Object Detection_20170331](https://www.slideshare.net/JihongKang/recent-progress-on-object-detection20170331)


# RCNN
Approaches using RCNN-trained models in multi-stage pipelines (first detecting object boundaries and then performing identification) 
 - rather slow and not suited for real time processing. 

The drawback of this approach is mainly its __speed__, both during the training and during the actual testing while object detection was performed. 
    - eg. VGG16, the training process for a standard RCNN takes 2.5 GPU-days for the 5k images and requires hundreds of GB of storage. Detecting objects at test-time takes 47s/image using a GPU. This is mainly caused by performing a forward pass on the convolutional network for each object proposal, without sharing the computation.

# Fast R-CBB

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

- [Recognition using Regions]