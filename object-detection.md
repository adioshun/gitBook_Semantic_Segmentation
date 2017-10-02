## Object Detection 

![](http://i.imgur.com/9xApnEN.png)

![](http://i.imgur.com/8V91Ouw.png)

- Region 기반 딥러닝 : R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN

- Grid 기반 딥러닝 : YOLO, DetectNet
    -  YOLO는 이미지 내의 bounding box와 class probability를 single regression problem으로 간주하여, 이미지를 한 번 보는 것으로 오브젝트의 종류와 위치를 추측합니다
    - 아래와 같이 single convolutional network를 통해 multiple bounding box에 대한 class probability를 계산하는 방식을 취합니다.


![](https://i.imgur.com/ZFg9tdp.png)

|년도|알고리즘|링크|입력|출력|특징|
|-|-|-|-|-|-|
|2014|R-CNN|[논문](https://arxiv.org/abs/1311.2524)|Image|Bounding boxes + labels for each object in the image.|AlexNet, 'Selective Search'사용 |
|2015|Fast R-CNN|[논문](https://arxiv.org/abs/1504.08083)|Images with region proposals.|Object classifications |Speeding up and Simplifying R-CNN, RoI Pooling|
|2016|Faster R-CNN|[논문](https://arxiv.org/abs/1506.01497)| CNN Feature Map.|A bounding box per anchor|MS, Region Proposal|
||YOLO|||||
||SSD||||Faster R-CNN + YOLO|

> 2017년 [Mask R-CNN](https://arxiv.org/abs/1703.06870)이 발표 되었지만 Segmentation 분야여서 포함 안함 


# 참고 자료 

- [Selective Search for Object Recognition](https://www.koen.me/research/pub/uijlings-ijcv2013-draft.pdf)

- Recognition using Regions

- [Regression Methods for Localization](https://bfeba431-a-62cb3a1a-s-sites.googlegroups.com/site/deeplearningcvpr2014/RegressionMethodsforLocalization.pdf?attachauth=ANoY7cpf41j03XW6YUpHg5L5_LgNhz6C05lpU58CkgQixIXesT0WOK6HU3CVi5x8t83aWcvYkvrUIpZ80rXYI8Hnlfk-wFdcay_DWW4c9ww5KXDADhcyhMiCDOv3AnNkhmuQDLFWCxyjY--VParh1WCIVUIOvtj4NW_UPc2zz0I_b9ovWkK-_qEio3oAY29Z6cyzK4Co60biKGRrc_3WfXxJgdq0Zq7pPnopAAHdEFpU9bv360H-EeW88n-h--8fyCQhJsG7-Pm-&attredirects=0): ppt

- J. Long, “Fully convolutional networks for semantic segmentation,” in IEEE CVPR 2015. :[Github](https://github.com/shelhamer/fcn.berkeleyvision.org), [한글분석](http://www.whydsp.org/317), [Arxiv](https://arxiv.org/abs/1605.06211), [ppt](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-pixels.pdf)

- K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition,
CVPR 2016. (R-CNNs are based on ResNets)



|Method|VOC2007|VOC2010|VOC2012|ILSVRC 2013|MSCOCO 2015|Speed|
|--- |--- |--- |--- |--- |--- |--- |
|OverFeat|-|-|-|24.3%|-|-|
|R-CNN (AlexNet)|58.5%|53.7%|53.3%|31.4%|-|-|
|R-CNN (VGG16)|66.0%|-|-|-|-|-|
|SPP_net(ZF-5)|54.2%(1-model), 60.9%(2-model)|-|-|31.84%(1-model), 35.11%(6-model)|-|-|
|DeepID-Net|64.1%|-|-|50.3%|-|-|
|NoC|73.3%|-|68.8%|-|-|-|
|Fast-RCNN (VGG16)|70.0%|68.8%|68.4%|-|19.7%(@[0.5-0.95]), 35.9%(@0.5)|-|
|MR-CNN|78.2%|-|73.9%|-|-|-|
|Faster-RCNN (VGG16)|78.8%|-|75.9%|-|21.9%(@[0.5-0.95]), 42.7%(@0.5)|198ms|
|Faster-RCNN (ResNet-101)|85.6%|-|83.8%|-|37.4%(@[0.5-0.95]), 59.0%(@0.5)|-|
|SSD300 (VGG16)|72.1%|-|-|-|-|58 fps|
|SSD500 (VGG16)|75.1%|-|-|-|-|23 fps|
|ION|79.2%|-|76.4%|-|-|-|
|CRAFT|75.7%|-|71.3%|48.5%|-|-|
|OHEM|78.9%|-|76.3%|-|25.5%(@[0.5-0.95]), 45.9%(@0.5)|-|
|R-FCN (ResNet-50)|77.4%|-|-|-|-|0.12sec(K40), 0.09sec(TitianX)|
|R-FCN (ResNet-101)|79.5%|-|-|-|-|0.17sec(K40), 0.12sec(TitianX)|
|R-FCN (ResNet-101),multi sc train|83.6%|-|82.0%|-|31.5%(@[0.5-0.95]), 53.2%(@0.5)|-|
|PVANet 9.0|81.8%|-|82.5%|-|-|750ms(CPU), 46ms(TitianX)|

> [출처](https://github.com/Smorodov/Deep-learning-object-detection-links./blob/master/readme.md) : 추후 살펴 보기, [블로그](https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html)

--- 

# 성능향상 기법 

- Hard negative mining을 이용한 2~3% 성능 향상 방법들에 대한 서베이 논문 
    - [Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/abs/1604.03540)