|논문명/저자/소속|Fast R-CNN|
|-|-|
|저자(소속)|Ross Girshick(MS)|
|학회/년도|ICCV 2015, [논문](https://arxiv.org/abs/1504.08083)|
|키워드||
|참고|[코드_Python](https://github.com/rbgirshick/fast-rcnn)|


# Fast Region-based Convolutional Network method (Fast R-CNN)

## 1. Introduction

Recently, deep ConvNets [14, 16] have significantly improved image classification [14] and object detection [9, 19]accuracy. 

Compared to image classification, object detection is a more challenging task that requires more complex methods to solve. 

Due to this complexity, current approaches (e.g., [9, 11, 19, 25]) train models in multi-stage pipelines that are slow and inelegant.

> DCN은 이미지 분류(image classification)과 물체탐지(object detection)에 사용된다. 물체 탐지기술은 이미지 분류 보다 더 복작하여 그 속도도 느리다. 

Complexity arises because detection requires the accurate localization of objects, creating two primary challenges. 
- First, numerous candidate object locations (often called “proposals”) must be processed. 

- Second, these candidates provide only rough localization that must be refined to achieve precise localization. 

Solutions to these problems often compromise speed, accuracy, or simplicity.

> 복잡도 역시 물체의 Localization 정확도로 두가지 챌린지에 직면한다. 
> - Proposal이라고 불리우는 다수의 물체 예상 위치에 대한 처리 
> - 예상 위치를 재처리 하여 위치 정확도 올리기 

In this paper, we streamline the training process for state of-the-art ConvNet-based object detectors [9, 11]. We propose a single-stage training algorithm that jointly learns to classify object proposals and refine their spatial locations.
- The resulting method can train a very deep detection network (VGG16 [20]) 9× faster than R-CNN [9] and 3×faster than SPPnet [11]. 
- At runtime, the detection network processes images in 0.3s (excluding object proposal time) while achieving top accuracy on PASCAL VOC 2012 [7]with a mAP of 66% (vs. 62% for R-CNN)

> 본 논문에서는 공동으로 object proposals 분류와 위치 재처리를 하는 single-stage training algorithm을 제안한다. 

### 1.1. R-CNN and SPPnet

#### A. R-CNN

The Region-based Convolutional Network method (RCNN) [9] achieves excellent object detection accuracy by using a deep ConvNet to classify object proposals. 
> R-CNN은 ConvNet을 이용하여 object proposals 분류 하는 방법ㅇ르 사용하여 좋은 성능을 보였다. 

하지만 큰 단점을 가지고 있다. R-CNN,however, has notable drawbacks:

##### 가. Training is a multi-stage pipeline. 
- R-CNN first finetunes a ConvNet on object proposals using log loss.Then, it fits SVMs to ConvNet features. 
- These SVMsact as object detectors, replacing the softmax classifier learnt by fine-tuning. 
In the third training stage,bounding-box regressors are learned.

##### 나. Training is expensive in space and time. 
- For SVMand bounding-box regressor training, features are extracted from each object proposal in each image andwritten to disk. 
- With very deep networks, such asVGG16, this process takes 2.5 GPU-days for the 5kimages of the VOC07 trainval set. 
These features require hundreds of gigabytes of storage.

##### 다. Object detection is slow. 
- At test-time, features areextracted from each object proposal in each test image.Detection with VGG16 takes 47s / image (on a GPU).


R-CNN is slow because it performs a ConvNet forward pass for each object proposal, without sharing computation. Spatial pyramid pooling networks (SPPnets) [11] were proposed to speed up R-CNN by sharing computation. 
> R-CNN은 연산을 공유 하지 않아서 속도가 느리다 이를 해결 하기 위해서 SPPnets이 제안 되었다. 

#### B. SPPnet

The SPPnet method computes a convolutional feature map for the entire input image and then classifies each object proposal using a feature vector extracted from the shared feature map. 
- Features are extracted for a proposal by max pooling the portion of the feature map inside the proposal into a fixed-size output (e.g., 6 × 6). 
- Multiple output sizes are pooled and then concatenated as in spatial pyramid pooling [15]. 

> SPPnet은 입력 이미지 전체에 대한 `convolutional feature map`을 계산한다. 그리고 나서 feature vector를 이용하여  각 object proposal 분류 한다. 

SPPnet accelerates R-CNN by 10 to 100× at test time. Training time is also reduced by 3× due to faster proposal feature extraction

SPPnet역시 단점을 가지고 있다. SPPnet also has notable drawbacks. 
- Like R-CNN, training is a multi-stage pipeline that involves extracting features, fine-tuning a network with log loss, training SVMs, and finally fitting bounding-box regressors. 
Features are also written to disk. 

- But unlike R-CNN, the fine-tuning algorithm proposed in [11] cannot update the convolutional layers that precede the spatial pyramid pooling. 

Unsurprisingly, this limitation (fixed convolutional layers) limits the accuracy of very deep networks.

### 1.2 Contributions
We propose a new training algorithm that fixes the disadvantages of R-CNN and SPPnet, while improving on their speed and accuracy. We call this method Fast R-CNN because it’s comparatively fast to train and test. 

The Fast RCNN method has several advantages:
- Higher detection quality ([mAP][ref_mAP]) than R-CNN, SPPnet
- Training is single-stage, using a multi-task loss
- Training can update all network layers
- No disk storage is required for feature caching 

Fast R-CNN is written in Python and C++ (Caffe[13]) and is available under the open-source MIT License at https://github.com/rbgirshick/fast-rcnn.


## 2. Fast R-CNN architecture and training

![](http://i.imgur.com/dx4kWU1.png)

Fig. 1 illustrates the Fast R-CNN architecture. 

- 입력 : A FastR-CNN network takes as input an entire image and a set of object proposals. 

- The network first processes the whole image with several convolutional (conv) and max pooling layers to produce a conv feature map. 

- Then, for each object proposal a region of interest (RoI) pooling layer extracts a fixed-length feature vector from the feature map.

- Each feature vector is fed into a sequence of fully connected(fc) layers that finally branch into two sibling output layers: 
  - one that produces softmax probability estimates over $$K$$ object classes plus a catch-all “background” class and 
  - another layer that outputs four real-valued numbers for each of the K object classes. 
    - Each set of 4 values encodes refined bounding-box positions for one of the K classes.

```
그림 1은 Fast R-CNN의 구조도이다. 

1. 입력으로 [전체 이미지] + [set of object proposal]을 받는다. 
2. 이미지를 conv feature map으로 만든다. (convolutional and max pooling Layer 이용)
3. each object proposal의 conv feature map에서 feature vector를 추출 한다. (RoI pooling layer이용)
4. Each feature vector는 sequence of fully connected(fc) layers로 전달된다. FC는 다시 2개의 레이어로 분리 된다. 
  - outputs softmax probability estimates
  - outputs four real-valued numbers

```


---
[ref_mAP]:http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf

  Mean Average Precision, [[The PASCAL Visual Object Classes (VOC) Challenge]](http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf)
- To calculate it for Object Detection, you calculate the average precision for each class in your data based on your model predictions. Average precision is related to the area under the precision-recall curve for a class. Then Taking the mean of these average individual-class-precision gives you the Mean Average Precision.