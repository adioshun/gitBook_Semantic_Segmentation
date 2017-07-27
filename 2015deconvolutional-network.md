|논문명|Learning deconvolutional network for semantic segmentation|
|-|-|
|저자(소속)|노현우 (포항공대)|
|학회/년도|2015, [논문](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Noh_Learning_Deconvolution_Network_ICCV_2015_paper.pdf)|
|키워드| |
|참고|[라온피플](http://laonple.blog.me/221019319607)|


# Deconvolutional network

## 1. 개요 

FCN에서 "conv+pooling"을 거치면서 해상도가 작아지는 문제를 반복적인 up-convolution(deconvolution)을 통해 해결을 시도하였다

FCN : classification용 네트워크 + up-sampling 로직 + skip layer 개념 = semantic segmentation 적용가능

### 1.1 FCN의 문제점 
![](http://i.imgur.com/0ZhEBwW.png)

#### A. Missing labels due to small object size 

사전에 미리 정한 receptive field를 사용하기 때문에 너무 작은 object가 무시되거나 엉뚱하게 인식될 수 있으며, 큰 물체를 여러 개의 작은 물체로 인식하거나 일관되지 않은 결과가 나올 수도 있다. 

#### B. Inconsistent labels due to large object size 
 여러 단의 "conv+pooling"을 거치면서 해상도가 줄어들고, 줄어든 해상도를 다시 upsampling을 통해 복원하는 방식을 사용하기 때문에, detail이 사라지거나 과도하게 smoothing 효과에 의해 결과가 아주 정밀하지 못하다.
 

### 1.2 해결책 

![](http://i.imgur.com/i4TYK57.png)

원인 정의 : 픽셀 단위의 조밀한 예측을 위해 upsampling과 여러 개의 후반부 layer의 conv feature를 합치는 방식

아이디어 : convolutional network에 대칭이 되는 deconvolutional network을 추가

효과 : upsampling 해상도의 문제를 해결


### 1.3 deconvolutional network 효과

- VGG16을 기반, ZFNet의 switch variable 개념 응용 (max-pooling의 위치를 기억 하기 위한 방법)

![](http://i.imgur.com/UedZLVM.png)


- decovolutional network의 각 layer에서 얻어지는 activation을 시각화 시킨 그림
![](http://i.imgur.com/COibKSB.png)

단순한 decovolution이나 upsampling을 사용하는 대신에 coarse-to-fine deconvolution 망을 구성함으로써 보다 정교한 예측이 가능함을 확인할 수 있다. 

Unpooling을 통해 가장 강한 activation을 보이는 위치를 정확하게 복원함에 따라 특정 개체의 더 특화된(논문의 표현은 exampling-specific) 구조를 얻어낼 수 있고, deconvolution을 통해 개체의 class에 특화된 (class-specific) 구조를 추출해 낼 수 있게 된다.

#### 1.4 FCN VS. Deconvolutional network 

![](http://i.imgur.com/pO1Ypie.png)

FCN은 전체적인 형태를 추출하는 것에 적합
Deconvolutional network은 좀 더 정밀한 segmentation에서 좋은 특징 보임


이 둘을 섞어 사용을 하면 더 좋은 결과를 얻을 수 있다

- FCN과 Deconvolutioal network의 결과의 평균

- 결과에 추가적으로 CRF(Conditional Random Field)를 적용

## 2. 구조

## 3. 특징

## 4. 학습/테스트 
효율적인 학습을 위해 2단계의 학습법을 사용하였다.
​
- 1단계
 - 먼저 쉬운 이미지를 이용하여 학습을 시키는데, 가급적이면 학습을 시킬 대상이 영상의 가운데에 위치할 수 있도록 하였고, 크기도 대체적으로 변화가 작도록 설정을 하였다. 
  - 1차 학습은 pre-training의 용도로 생각할 수 있으며, data augmentation을 거친 약 20만장의 이미지를 사용하였다.
​
- 2단계
 - 270만장의 이미지를 사용하였으며, 좀 더 다양한 경우에 대응할 수 있도록 다양한 크기 및 위치에 대응이 가능할 수 있도록 하였다.
 
 



