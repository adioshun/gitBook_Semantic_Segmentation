|논문명|Fully Convolutional Networks for Semantic Segmentation|
|-|-|
|저자(소속)|Jonathan Long, Evan Shelhamer*, and Trevor Darrell(버클리)|
|학회/년도|CVPR 2015, [논문](https://arxiv.org/pdf/1605.06211.pdf)|
|키워드| |
|참고|[모연](http://www.whydsp.org/317), [리뷰](http://warmspringwinds.github.io/tensorflow/tf-slim/2017/01/23/fully-convolutional-networks-(fcns)-for-image-segmentation/)|
|코드|[caffe](https://github.com/shelhamer/fcn.berkeleyvision.org), [pyTorch](https://github.com/wkentaro/pytorch-fcn)|


# FCN

Fully Convolutional Network $$\ne$$ Fully Connected Network

## 1. 개요 

> Semantic segmentation은 영상속에 무엇(what)이 있는지를 확인하는 것(semantic)뿐만 아니라 어느 위치(where)에 있는지(location)까지 정확하게 파악을 해줘야 한다. 
    
### 1.1 문제 정의 

Classification 기반 망을 semantic segmentation에 적용할 때의 문제점

#### A. conv+pooling 문제 

- Classification 망 : 대상의 존재 여부에 집중, conv+pooling을 통해 강한 Feature들 추출 목적, detail보다는 global한 것에 집중
    - 강한 Feature : 많은 변화(conv+pooling)에 영향을 받지 않는 `공간적인 불변성(spatial invariance)`
     

- semantic segmentation 망 : 픽셀 단위의 조밀한 예측이 필요, 분류망은 pooling등을 통해 feature-map의 크기가 줄어들기 때문에 detail 정보를 얻는데 어려움

#### B. fully connected layer 문제 
- 기존 Classification 네트워크 (AlexNet, VGGNet, GoogLeNet)는 마지막에 분류를 위해  fully connected layer 사용 

- fully connected layer를 사용하면 분류는 되지만 위치 정보가 사라진다. 

- segmentation에서 사용 불가


> FCN에서는 `B. fully connected layer 문제`문제에 집중 

### 1.2 해결책 

- 기본 아이디어 : fully connected layer를  1x1 convolution으로 간주(convolutionization) 

![](http://i.imgur.com/Gc07zsQ.png)


위치 정보가 남아 있기 때문에 오른쪽의 heatmap 그림에서 알 수 있는 것처럼, 고양이에 해당하는 위치﻿의 score 값들이 높은 값으로 나오게 된다

> 위치는 알수 있는데 그럼 분류를 못하는 것인가??? 

### 1.3 장점 

- 위치 정보 유지 

- 입력 이미지 제약 없음 
  - 입력 이미지 제약은 fully connected layer에 맞추기 위해서 존재 하였음
  - 모든 network가 convolutional network으로 구성이 되기 때문에 더 이상 입력 이미지의 크기 제한을 받지 않게 된다

- 속도 증가 
  - patch 단위로 영상을 처리하는 것이 아니라, 전체 영상을 한꺼번에 처리를 할 수 있어서 겹치는 부분에 대한 연산 절감 효과를 얻을 수 있게 되어, 속도가 빨라지게 된다
  - eg.   Fast R-CNN이나 Faster R-CNN에서 이 아이디어 활용 
  
### 1.4 문제점 

![](http://i.imgur.com/3cXXYgr.png)

그런데 여러 단계의 (convolution + pooling)을 거치게 되면, feature-map의 크기가 줄어들게 된다. 

픽셀 단위로 예측(1x1 convolution)을 하려면, 줄어든 feature-map의 결과를 다시 키우는 과정(Upsampling/Deconvolution)을 거쳐야 한다. 




## 2. Upsampling 하는 방법 (가능한 손실을 보정 하면서 )


### 2.1 가장 간단한 방법 : bilinear interpolation

> bilinear interpolation : 두 점사이의 값을 추정 하는것 [[참고]](http://darkpgmr.tistory.com/117)

하지만 end-to-end 학습의 관점에서는 고정된 값을 사용하는 것이 아니라 학습을 통해서 결정하는 편이 좋다. 논문에서는 backward convolution, 즉 deconvolution을 사용하며, deconvolution에 사용하는 필터의 계수는 학습을 통해서 결정이 되도록 하였다. 이렇게 되면, 경우에 따라서 bilinear 한 필터를 학습할 수도 있고 non-linear upsampling도 가능하게 된다.


### 2.2  제안 방법 : Skip layer

기본 개념: (convolution + pooling)의 과정을 여러 단계를 거치면서 feature-map의 크기가 너무 작아지면 detail한 부분이 많이 사라지게 되기 때문에 **최종 과정**보다 **앞선 결과**를 사용하여 detail을 보강하자는 것이다.

- 최종 과정 : 1/32크기의 feature(즉 score)
- 앞선 결과 : 1/16과 1/8크기의 feature(즉 score)

> 논문에서는 이를 `deep jet`라 부름,  이전 layer는 마지막 layer보다 세밀한 feature를 갖고 있기 때문에 이것을 합하면 보다 정교한 예측이 가능

![](http://i.imgur.com/jAPbPDb.png)








앞선 결과(1/16과 1/8크기)를 이용할수 있는 방법 : 이후 작업을 수행 하지 않고 빼놓음
  -  "skip layer" 혹은 "skip connection라고 부름 











![](http://i.imgur.com/jCHUmUv.png)

![](http://i.imgur.com/s7tNztd.png)

- FCN-32s : 1/32에서 32배만큼 upsample한 결과
- FCN-16s : pool5의 결과를 2배 upsample한 것과 pool4의 결과를 합치고 다시 그 결과를 16배 upsample하면 되고, 
- FCN-8s : FCN-16s의 중간 결과를 2배 upsample한 결과와 pool3에서의 예측을 합친 것을 8배 upsample 하는 식이다.
​

###### FCN-8s 상세 
  - 실제로 FCN의 결과는 주로 FCN-8s의 결과를 사용
  
![](http://i.imgur.com/V7VkrKh.png)


### 2.3 기타 방법 

- shift-and-stitch 방식

- Deconvolutional 방식 : Unpooling + Deconvolution 수행 

- Dilated convolution 방식 : pool4,5는 제거 + Conv 4,5를 Dilated Convolution으로 변경 




