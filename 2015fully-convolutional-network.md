|논문명|Fully Convolutional Networks for Semantic Segmentation|
|-|-|
|저자(소속)|Jonathan Long, Evan Shelhamer*, and Trevor Darrell(버클리)|
|학회/년도|CVPR 2015, [논문](https://arxiv.org/pdf/1605.06211.pdf)|
|키워드| |
|참고|[모연](http://www.whydsp.org/317), [코드](https://github.com/shelhamer/fcn.berkeleyvision.org)|


# FCN

Fully Convolutional Network $$\ne$$ Fully Connected Network

## 1. 개요 

> Semantic segmentation은 영상속에 무엇(what)이 있는지를 확인하는 것(semantic)뿐만 아니라 어느 위치(where)에 있는지(location)까지 정확하게 파악을 해줘야 한다. 
    
### 1.1 문제 정의 

- 기존 Classification 네트워크 (AlexNet, VGGNet, GoogLeNet)는 마지막에 분류를 위해  fully connected layer 사용 

- fully connected layer를 사용하면 분류는 되지만 위치 정보가 사라진다. 

- segmentation에서 사용 불가

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
  



## 2. 특징 



### 2.1  skip connection



![](http://i.imgur.com/3cXXYgr.png)

그런데 여러 단계의 (convolution + pooling)을 거치게 되면, feature-map의 크기가 줄어들게 된다. 

픽셀 단위로 예측(1x1 convolution)을 하려면, 줄어든 feature-map의 결과를 다시 키우는 과정(skip connection)을 거쳐야 한다.

> bilinear interpolation : 두 점사이의 값을 추정 하는것 [[참고]](http://darkpgmr.tistory.com/117)


더 자세히 살펴 보기 : [라온피플 블로그](http://laonple.blog.me/220935916241)


![](http://i.imgur.com/jCHUmUv.png)

![](http://i.imgur.com/s7tNztd.png)

FCN-32s : 1/32에서 32배만큼 upsample한 결과
FCN-16s : pool5의 결과를 2배 upsample한 것과 pool4의 결과를 합치고 다시 그 결과를 16배 upsample하면 되고, 
FCN-8s : FCN-16s의 중간 결과를 2배 upsample한 결과와 pool3에서의 예측을 합친 것을 8배 upsample 하는 식이다.
​
![](http://i.imgur.com/V7VkrKh.png)
