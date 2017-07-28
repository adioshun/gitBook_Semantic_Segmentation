|논문명|DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs|
|-|-|
|저자(소속)|Liang-Chieh Chen (구글)|
|학회/년도|2016 [논문](https://arxiv.org/pdf/1606.00915.pdf)|
|키워드| |
|참고|[라온피플](http://laonple.blog.me/221000648527)|


# DeepLab v2

2015: DeepLab v1
2015: dilated convolution, DeepLab v1의 단점 개선
2016: DeepLab v1, multiple-scale에 대한 처리 방법 개선, dilated convolution보다 성능 좋음 



## 1. 개요 

semantic segmentation = DCNN(deep convolutional neural networks) + atrous convolution + fully connected CRF

### 1.1 Classification 기반 망을 semantic segmentation에 적용할 때의 문제점

- Classification 망 : 대상의 존재 여부에 집중, conv+pooling을 통해 강한 Feature들 추출 목적, detail보다는 global한 것에 집중

- semantic segmentation 망 : 픽셀 단위의 조밀한 예측이 필요, 분류망은 pooling등을 통해 feature-map의 크기가 줄어들기 때문에 detail 정보를 얻는데 어려움

### 1.2 detail 정보를 얻는데 어려운 문제 해결책

- FCN : 
    - skip layer를 사용하여 1/8, 1/16, 및 1/32 결과를 결합하여 detail이 줄어드는 문제를 보강
    - bilinear interpolation을 이용해 원 영상을 복원

- dilated convolution/DeepLab : 
    - 망의 뒷 단에 있는 2개의 pooling layer를 제거하여 1/8크기의 feature-map을 얻고 
    - dilated convolution 혹은 atrous convolution을 사용하여 receptive field를 확장시키는 효과를 얻었으며
        - 이렇게 1/8 크기까지만 줄이는 방법을 사용하여 detail이 사라지는 것을 커버
    - bilinear interpolation을 이용해 원 영상을 복원

### 1.3 1/8까지만 사용 시 문제점

- Receptive field가 충분히 크지 않아 다양한 scale에 대응이 어렵다.

- 1/8정보를 bilinear interpolation을 통해서 원 영상의 크기로 키우면, 1/32 크기를 확장한 것보다 detail이 살아 있기는 하지만, 여전히 정교함이 떨어진다.

## 2. 구조

### 2.1 Atrous Convolution

Atrous Convolution : 보다 넓은 scale을 보기 위해 중간에 hole(zero)을 채워 넣고 convolution을 수행하는 것을 말한다
    - 원래 웨이브릿(wavelet)을 이용한 신호 분석에 사용되던 방식이며

![](http://i.imgur.com/7IXynCh.png)

- (a)는 기본적인 convolution이며 인접 데이터를 이용해 kernel의 크기가 3인 convolution의 예이다.

- (b)는 확장 계수(k)가 2인 경우로 인접한 데이터를 이용하는 것이 아니라 중간에 hole이 1개씩 들어오는 점이 다르며, 똑 같은 kernel 3을 사용하더라도 대응하는 영역의 크기가 커졌음을 확인할 수 있다.

atrous convolution(dilated convolution)을 사용하면 kernel 크기는 동일하게 유지하기 때문에 연산량은 동일하지만, receptive field의 크기가 커지는(확장되는) 효과를 얻을 수 있게 된다

즉, 
- 기존 : pooling 후 동일한 크기의 convolution을 수행하면, 자연스럽게 receptive field가 넓어지는데
- 제안 : pooling layer를 제거하여 receptive field가 넓어지지 않는 문제를 
    - atrous convolution을 사용하여 더 넓은 receptive field를 볼 수 있도록 하였다. 


### 2.2 Atrous convolution 및 Bilinear interpolation

![](http://i.imgur.com/lBqiyxo.png)

1. Atrous convolution은 receptive field 확대를 통해 특징을 찾는 범위를 넓게 해주기 때문에 전체 영상으로 찾는 범위를 확대하면 좋음 

2. Atrous convolution 문제점 : 단계적으로 수행을 해줘야 하기 때문에 연산량이 많이 소요될 수 있다.

3. 해결책 : 그래서 적정한 선에서 나머지는 bilinear interpolation을 선택하였다.

  
4. bilinear interpolation 문제점 : 정확하게 비행기를 픽셀단위까지 정교하게 segmentation을 한다는 것이 불가능

5. bilinear interpolation 해결책 : 뒷부분은 CRF(Conditional Random Field)를 이용하여 post-processing을 수행


### 2.3 
