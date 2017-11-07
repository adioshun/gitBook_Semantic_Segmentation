|논문명 | |
| --- | --- |
| 저자\(소속\) | \(\) |
| 학회/년도 | IROS 2015, [논문]() |
| 키워드 | |
| 데이터셋(센서)/모델 | |
| 관련연구||
| 참고 | |
| 코드 | |





# AlexNet

> ImageNet Classification with Deep Convolutional Neural Networks, 2012

## 1. 개요

컴퓨터 비전 분야에서 CNN이 널리 적용되도록 한 계기가 된 구조로,

CNN 분야에서 유명한 Krizhevsky와 Hinton 등에 의해 개발이 되었다.

2012년 ImageNet ILSVRC 대회에서

2위와 큰 성능차(AlexnNet의 에러율은 16%이고 2위의 에러율은 26%)를 보이며 우승한 것으로 유명하다.

## 2. 네트워크 구조  

AlexNet의 구조는 LeNet과 유사하지만,

보통 convolutional layer 다음에 pooling(sub-sampling) layer가 오는 기본 구조와 달리,

convolutional layer 바로 뒤에 convolutional layer가 온 점이 특이하다.

![](http://i.imgur.com/LJPDB41.png)

위아래로 나누어져 있는것은 2개의 GPU를 기반으로 한 병렬 구조여서 그렇다. 기본구조는 LeNet5와 크게 다르지 않다.
- GPU-1에서는 주로 컬러와 상관없는 정보를 추출하기 위한 kernel이 학습이 되고,
- GPU-2에서는 주로 color에 관련된 정보를 추출하기 위한 kernel이 학습이 된다. 

## 3. 특징

### 3.1 ReLU
sigmoid , tanh은 학습 속도가 느리다. Alexnet처럼 깊은 망에서는 사용하기 어렵다. 

ReLU의 경우 sigmoid , tanh대비 약 6배 속도 향상을 가져 온다. 

또한, normalization 불필요 

> ReLU의 장점은 속도 향상 뿐인가?

### 3.2 overlapped pooling

overlapped pooling = Stride로 2 사용(Pooling효과) + Pooling Window로 3 x 3 윈도우 사용 

> Stride를 1 이상의 값을 주면 Pooling처럼 크기가 줄어 드는 효과가 있음 

### 3.3 response normalization (??)

> AlextNet 논문의 3.3 참고

ReLU의 장점중 하나는 입력의 normalization이 필요가 없다는 점이다. 하지만 ReLU는 입력에 비례하여 증가 하는 특징이 있다. 

ReLU는 포화를 막기 위한 입력 정규화를 필요로 하지 않음
그러나, ReLU 다음에 적용되는 지역 정규화 방식이 일반화를 도움
결과적으로 인접한 커널들에 걸쳐 정규화됨.

![](http://i.imgur.com/QTcyx0L.png)
- i는 커널 인덱스
- x,y 는 위치 좌표
- a는 뉴런의 출력


AlexNet은 첫번째와 두번째 convolution을 거친 결과에 대하여 ReLU를 수행하고, max pooling을 수행하기에 앞서 response normalization을 수행하였다.

![](http://i.imgur.com/hLbi2T8.png)


### 3.4 Data Augmentation (오버피팅 해결 책)
free parameter의 개수는 6000만개 수준이어서 오버피팅에 빠질 우려가 있음 이에 대한 해결책으로 Dropout사용 

#### A. Cropping

원 영상으로부터 무작위로 224x224 크기의 영상을 취하는 것

1장의 학습 영상으로부터 2048개의 다른 영상을 얻을 수 있게 되었다


#### B. RGB 채널의 값을 변화

학습 이미지의 RGB 픽셀 값에 대한 주성분 분석(PCA)를 수행하였으며, 거기에 평균은 0, 표준편차는 0.1 크기를 갖는 랜덤 변수를 곱하고 그것을 원래 픽셀 값에 더해주는 방식

### 3.5 dropout (오버피팅 해결 책)

AlexNet에서는 fully connected layer의 처음 2개 layer에 대해서만 적용을 하였다.

또한 dropout의 비율은 50%를 사용하였다.




### 3.6 2개의 GPU 사용

> Krizhevsky, "One weird trick for parallelizing convolutional neural networks",  2014


![](http://i.imgur.com/R7jZ5xQ.png)

요즘의 Deep CNN은 크게 2개로 구성이 되었다고 볼 수 있다.
- convolutional layer : 전체 연산량의 90~95%를 차지하지만, free parameter의 개수는 5% 정도 
    - Data Parallelism, filter 연산(=matrix multiplication) 수행 
- fully connected layer : 전체 연산량의 5~10%를 차지하지만, free parameter의 개수는 95% 정도
    - Model Parallelism 

