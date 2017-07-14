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
### 2.1 The RoI pooling layer
The RoI pooling layer uses max pooling to convert the features inside any valid region of interest into a small feature map with a fixed spatial extent of H × W (e.g., 7 × 7), where H and W are layer hyper-parameters that are independent of any particular RoI. 

> RoI pooling layer는 max pooling을 사용하며, 유요한 RoI내 Features를 small feature map으로 변환 하는 일을 진행 한다. 

In this paper, an RoI is a rectangular window into a conv feature map. Each RoI is defined by a four-tuple (r, c, h, w) that specifies its top-left corner (r, c) and its height and width (h, w).
> 본논문에서 RoI는 사각형 창이며 위치정보를 가지고 있다. 

RoI max pooling works by dividing the h × w RoI window into an H × W grid of sub-windows of approximatesize h=H × w=W and then max-pooling the values in eachsub-window into the corresponding output grid cell. 

Pooling is applied independently to each feature map channel,as in standard max pooling. 

The RoI layer is simply the special-case of the spatial pyramid pooling layer used in SPPnets [11] in which there is only one pyramid level. 

We use the pooling sub-window calculation given in [11].

### 2.2. Initializing from pre-trained networks

We experiment with three pre-trained ImageNet [4] networks, each with five max pooling layers and between five and thirteen conv layers (see Section 4.1 for network details). 
> ImageNet 네트워크로 실험 하였다. 

When a pre-trained network initializes a Fast R-CNN network, it undergoes three transformations.
- First, the last max pooling layer is replaced by a RoI pooling layer that is configured by setting H and W to be compatible with the net’s first fully connected layer (e.g.,H = W = 7 for VGG16).
- Second, the network’s last fully connected layer and softmax (which were trained for 1000-way ImageNet classification) are replaced with the two sibling layers described earlier (a fully connected layer and softmax over K + 1 categories and category-specific bounding-box regressors).
- Third, the network is modified to take two data inputs: a list of images and a list of RoIs in those images

변경 내용 

|ImageNet |Fast R-CNN network|
|-|-|
|last max pooling layer|RoI pooling layer|
|last fully connected layer + softmax  |two sibling layers|
||take two data inputs<br>- a list of images<br>- list of RoIs |

### 2.3. Fine-tuning for detection
Training all network weights with back-propagation is an important capability of Fast R-CNN. 
> 모든 가중치를 back-propagation로 학습하는것도 중요 기능 중의 하나이다. 

###### 기존 학습 알고리즘의 문제점 
First, let’s elucidate(설명) why SPPnet is unable to update weights below the spatial pyramid pooling layer.(먼저 SPPnet이 왜 가중치 학습이 안되는지 알아 보자)
- The root cause is that back-propagation through the SPP layer is highly inefficient when each training sample (i.e.RoI) comes from a different image, which is exactly how R-CNN and SPPnet networks are trained. 
- The inefficiency stems from the fact that each RoI may have a very large receptive field, often spanning the entire input image. 
- Since the forward pass must process the entire receptive field, the training inputs are large (often the entire image).

> SPPnet은 가중치 학습이 안된다. 가장 큰 이유는 각 학습 샘플들이 서로 다른 이미지에서 들어 오면 백프로파게이션은 매우 비 효율적이게 된다. 이것은 SPPnet이나 이전 R-CNN이 학습했던 방법들이다.  (?????????????)

###### 제안하는 학습 알고리즘 : sampled hierarchically

We propose a more efficient training method that takes advantage of feature sharing during training. 
> feature sharing의 장점을 이용한 효율적인 학습 방법을 제안 한다. 

In Fast RCNN training, stochastic gradient descent (SGD) mini-batches are sampled hierarchically, 
- first by sampling N images and then 
- by sampling R/N RoIs from each image.
Critically, RoIs from the same image share computation and memory in the forward and backward passes. 
Making N small decreases mini-batch computation. 

For example, when using N = 2 and R = 128, the proposed training scheme is roughly 64× faster than sampling one RoI from 128 different images (i.e., the R-CNN and SPPnet strategy).
> 예를 들어 N = 2 and R = 128일때 제안하는 학습 알고르짐은 기존 대비 64배 빠르다. 

One concern over this strategy is it may cause slow training convergence because RoIs from the same image are correlated. This concern does not appear to be a practical issue and we achieve good results with N = 2 and R = 128 using fewer SGD iterations than R-CNN.
> 걱정되는 문제는 동일 이미지상의RoI는 연관되어 있으므로 학습시 convergence 될수 있을까 하는 걱정이 있다. 하지만, 실제로는 발생 하지 않았고 좋은 결과를 보였따. 

###### 제안하는 학습 알고리즘 : streamlined training process

In addition to hierarchical sampling, Fast R-CNN uses a streamlined training process with one fine-tuning stage that jointly optimizes a softmax classifier and bounding-box regressors, rather than training a softmax classifier, SVMs,and regressors in three separate stages [9, 11]. 
> R-CNN은 한번에 fine-tuning stage 를 하는 streamlined training process를 이용한다. 기존은 3-stage

The components of this procedure are described below.
- the loss, 
- mini-batch sampling strategy, 
- back-propagation through RoI pooling layers, and 
- SGD hyper-parameters

#### A. 

---
[ref_mAP]:http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf

  Mean Average Precision, [[The PASCAL Visual Object Classes (VOC) Challenge]](http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf)
- To calculate it for Object Detection, you calculate the average precision for each class in your data based on your model predictions. Average precision is related to the area under the precision-recall curve for a class. Then Taking the mean of these average individual-class-precision gives you the Mean Average Precision.