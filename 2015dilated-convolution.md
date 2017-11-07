|논문명|Multi-Scale Context Aggregation by Dilated Convolutions|
|-|-|
|저자(소속)|Fisher Yu (프린스톤대)|
|학회/년도|ICLR 2016 [논문](https://arxiv.org/abs/1511.07122)|
|키워드| |
|참고|[라온피플](http://laonple.blog.me/221019319607), [PR100](https://youtu.be/JiC78rUF4iI?t=9m33s)|


# Dilated Convolution

> Dilated :확대, 확장

## 1. 개요 

### 1.1 역사
Dilated convolution은 기존 부터 있던 개념 

- FCN 개발자들은 dilated convolution 대신에 skip layer와 upsampling 개념을 사용

- DeepLab 개발팀도 dilated convolution를 적용 (cf. Fisher Yu와는 다른 접근 방식)
    - 기본 개념은 wavelet decomposition 알고리즘에서 "atrous algorithm"이라는 이름으로 사용
    - 구별을 위해 atrous convolution이라고 명명

### 1.2 정의

Dilated convolution
- 기본적인 convolution과 유사하지만 빨간색 점의 위치에 있는 픽셀들만 이용하여 convolution을 수행하는 것이 다르다
    - 이렇게 사용하는 이유는 해상도의 손실 없이receptive field의 크기를 확장할 수 있기 때문
    
- atrous convolution이라고도 불리는 이유는 전체 receptive field에서 빨간색 점의 위치만 계수가 존재하고 나머지는 모두 0으로 채워지기 때문

> atrous: a trous는 구멍(hole) 이라는 뜻의 프랑스어

![](http://i.imgur.com/Zn8jAjF.png)

- (a)는 1-dilated convolution이라고 부르는데, 이것은 우리가 기존에 흔히 알고 있던 convolution과 동일하다. 

- (b)는 2-dilated convolution이라고 부르며, (b)의 빨간색 위치에 있는 점들만 convolution 연산에 사용하며, 나머지는 0으로 채운다. 
    - 이렇게 되면 receptive field의 크기가 7x7 영역으로 커지는 셈이 된다. 

- (c)는 4-dilated convolution이며, receptive field의 크기는 15x15로 커지게 된다.

### 1.3. 장점 

#### A. 파라미터의 개수가 고정되어 큰연산 불필요 
- 큰 receptive field를 취하려면, 파라미터의 개수가 많아야 하지만, 
- dilated convolution을 사용하면 receptive field는 커지지만 파라미터의 개수는 늘어나지 않기 때문에 연산량 관점에서 탁월한 효과를 얻을 수 있다.

위 그림의 (b)에서 receptive field는 7x7이기 때문에 normal filter로 구현을 한다면 필터의 파라미터의 개수는 49개가 필요하며, convolution이 CNN에서 가장 높은 연산량을 차지한다는 것을 고려하면 상당한 부담으로 작용한다. 

하지만 dilated convolution을 사용하면 49개중 빨간점에 해당하는 부분에만 파라미터가 있는 것이나 마찬가지고 나머지 40개는 모두 0으로 채워지기 때문에 연산량 부담이 3x3 filter를 처리하는 것과 같다.

#### B. receptive field의 크기가 커진다 = 다양한 scale에 대한 대응이 가능
dilation 계수를 조정하면 다양한 scale에 대한 대응이 가능해진다. 

다양한 scale에서의 정보를 끄집어내려면 넓은 receptive field를 볼 수 있어야 하는데 dilated convolution을 사용하면 별 어려움이 없이 이것이 가능해진다.

###### cf. 기존 방식
- receptive field 확장을 위해 pooling layer를 통해 크기를 줄인 후 convolution을 수행하는 방식을 취했다. 
- 기본적으로 pooling을 통해 크기가 줄었기 때문에 동일한 크기의 filter를 사용하더라도 CNN 망의 뒷단으로 갈수록 넓은 receptive field를 커버할 수 있게 된다.


![](http://i.imgur.com/yAKj2LP.png)

- 위쪽은 앞서 설명한 것처럼 down-sampling 수단이 적용이 된 경우이며, 픽셀 단위 예측을 위해 up-sampling을 통해 영상의 크기를 키운 경우이며, 

- 아래는 dilated convolution(atrous convolution)을 통하여 얻은 결과이다. 

## 2. 구조

### 2.1 Front-end 모듈
- 기존(FCN) : VGG-16 classification을 거의 그대로 사용
- 변경(Dilated convolution) : VGG-16 classification을 일부 수정하여 사용 


![](http://i.imgur.com/7P8LjmC.png)



#### A. pool4와 pool5는 제거하였다. 

- FCN에서는 pool4와 pool5를 그대로 두었기 때문에 feature-map의 크기가 1/32까지 작아지고 그런 이유로 인해, 좀 더 해상도가 높은 pool4와 pool3 결과를 사용하기 위해 skip layer라는 것을 두었다.

- dilated convolution은 pool4와 pool5를 제거함으로써 최종 feature-map의 크기는 원영상의1/32이 아니라 1/8수준까지만 작아졌기 때문에 upsampling을 통해 원영상 크기로 크게 만들더라도 상당한 detail이 살아 있게 된다.


#### B. conv5, 6에 dilated convolution적용

conv5와 conv6(fc6)에는 일반적인 convolution을 사용하는 대신에 conv5에는 2-dilated convolution을 적용하고, conv6에는 4-dilated convolution을 적용하였다..


결과적으로 skip layer도 없고 망도 더 간단해졌기 때문에, 연산 측면에서는 훨씬 가벼워졌다. 

### 2.2 Context 모듈 

목적 : 다중 scale의 context를 잘 추출해내기 위한 context 모듈도 개발

종류 :  Basic모듈 & Large 모듈
- Basic type은 feature-map의 개수가 동일하지만 
- Large type은 feature-map의 개수가 늘었다가 최종단만 feature-map의 개수가 원래의 feature-map 개수와 같아지도록 했다.

Context 모듈은 기본적으로 어떤 망이든 적용이 가능할 수 있도록 설계를 하였으며, 자신들의 Front-end 모듈 뒤에 Context 모듈을 배치

###### [참고] Context 모듈의 구성

![](http://i.imgur.com/4lFeQ4k.png)
전부 convolutional layer로만 구성이 된다. 
- C는 feature-map의 개수를 나타내고, 
- Dilation은 dilated convolution의 확장 계수이며, convolution만으로 구성이 되지만, 뒷단으로 갈수록 receptive field의 크기가 커지는 것을 확인할 수 있다.







