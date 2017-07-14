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