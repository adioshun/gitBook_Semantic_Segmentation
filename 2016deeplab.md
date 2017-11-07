|논문명|DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs|
|-|-|
|저자(소속)|Liang-Chieh Chen (구글)|
|학회/년도|2016 [논문](https://arxiv.org/pdf/1606.00915.pdf)|
|키워드| |
|참고|[라온피플](http://laonple.blog.me/221000648527), [PR100](https://www.youtube.com/watch?v=JiC78rUF4iI&feature=share)|


# DeepLab v2

![](https://i.imgur.com/h8SOkWp.png)



## 1. 개요 

2015: DeepLab v1[[1]](#1)
2015: dilated convolution, DeepLab v1의 단점 개선
2016: DeepLab v2[[2]](#2), multiple-scale에 대한 처리 방법 개선, dilated convolution보다 성능 좋음 

- DeepLab V1 = DCNN + atrous convolution + fully connected CRF
- DeepLab V2 = DCNN + atrous convolution + fully connected CRF + ASPP  


### 1.1 기존 방식의 문제점 

###### FCN
- skip layer를 사용하여 1/8, 1/16, 및 1/32 결과를 결합하여 detail이 줄어드는 문제를 보강
- bilinear interpolation을 이용해 원 영상을 복원

###### Dilated convolution/DeepLab 
- 망의 뒷 단에 있는 2개의 pooling layer를 제거하여 1/8크기의 feature-map을 얻고 
- dilated convolution 혹은 atrous convolution을 사용하여 receptive field를 확장시키는 효과를 얻었으며
    - 이렇게 1/8 크기까지만 줄이는 방법을 사용하여 detail이 사라지는 것을 커버
- bilinear interpolation을 이용해 원 영상을 복원

### 1.2 1/8까지만 사용 시 문제점

- Receptive field가 충분히 크지 않아 다양한 scale에 대응이 어렵다.

- 1/8정보를 bilinear interpolation을 통해서 원 영상의 크기로 키우면, 1/32 크기를 확장한 것보다 detail이 살아 있기는 하지만, 여전히 정교함이 떨어진다.
    - CRF(Conditional Random Field)를 사용한 후처리로 보정 가능 [2.4]에서 자세히 설명 

### 1.3 제안 방안 

- Atrous Convolution

- CRF(Conditional Random Field)

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

2. 전체 확대시 문제점 : 단계적으로 수행을 해줘야 하기 때문에 연산량이 많이 소요될 수 있다.

3. 해결책 : 그래서 적정한 선에서 나머지는 bilinear interpolation을 선택하였다.

  
4. bilinear interpolation 문제점 : 정확하게 비행기를 픽셀단위까지 정교하게 segmentation을 한다는 것이 불가능

5. bilinear interpolation 해결책 : 뒷부분은 CRF(Conditional Random Field)를 이용하여 post-processing을 수행


### 2.3 ASPP(Atrous Spatial Pyramid Pooling)

- 목적 : multi-scale에 더 잘 대응할 수 있도록 하기 위한 방안 

- 모티브 : SPPNet 논문에 나오는 Spatial Pyramid Pooling기법

- 방법 : “fc6” layer에서의 atrous convolution을 위한 확장 계수를 아래 그림과 같이 {6, 12, 18. 24}로 적용을 하고 그 결과를 취합하여 사용

- 원리 :  확장 계수를 6부터 24까지 변화시킴으로써 다양한 receptive field를 볼 수 있게 되었다

![](http://i.imgur.com/7cpCz2a.png)

실험에 따르면, 단순하게 확장 계수 r을 12로 고정(DeepLab V1)시키는 것보다. 이렇게 ASPP를 지원함으로써 약 1.7% 정도 성능을 개선할 수 있게 되었다.

![](http://i.imgur.com/UJb7H9X.png)
- (a)는 DeepLab V1의 구조이며, fc6의 계수를 12로 고정(ASPP를 미적용)
- (b)는 DeepLab V2의 구조이며, fc6의 계수를 {6. 12, 18, 24}로 설정(ASPP를 적용) = 1/7% 성능향상 

### 2.4 Fully Connected CRF (후처리 과정 기술)

> Philipp Krahenbuhl, "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potential", 2011, 스탠포드대
> - [참고: Inference in Fully Connected CRFs with Gaussian Edge Potentials](http://swoh.web.engr.illinois.edu/courses/IE598/handout/fall2016_slide15.pdf)
> - [참고: DeepLab–semantic image segmentation](http://www.cs.jhu.edu/~ayuille/JHUcourses/ProbabilisticModelsOfVisualCognition/Lecture20DeepNetwork2/Pages%20from%20DeepMindTalk.pdf)

- 기존의 좁은 범위(short-range) CRF는 segmentation을 수행한 뒤에 생기는 segmentation 잡음을 없애는 용도로 많이 사용 됨

- 여러 단계 "conv+pooling"을 거치면서 크기가 작아지고 그것을 upsampling을 통해 원영상 크기로 확대하기 때문에 이미 충분히 부드러운(smooth) 상태에 short-range CRF 적용시 결과가 더 나빠지게 된다. 

- 해결책 : 전체 픽셀을 모두 연결한(fully connected) CRF 방법 적용 

||특징|장/단점|
|-|-|-|
|Short-range CRF|![](http://i.imgur.com/IwPraWp.png)|local connection 정보만을 사용 <br> - Detail정보 얻을 수 없음|
|Fully Connected CRF|![](http://i.imgur.com/Tf7I4YJ.png)| Fully Connection 정보 사용 <br> - detail정보 얻을 수 있음<br> - 단점: 시간이 오래 걸림(MCMC사용시 36시간)|

기존 오랜연산시간의 문제점을 message passing을 사용한 iteration 방법을 적용하여 해결 
    - 일명 : mean field approximation 방법
    
> Mean field approximation이란, 물리학이나 확률이론에서 많이 사용되는 방법- 
> - 복잡한 모델을 설명하기 위해 더 간단한 모델을 선택하는 방식을 말한다. 
> - 수많은 변수들로 이루어진 복잡한 관계를 갖는 상황에서 특정 변수와 다른 변수들의 관계의 평균을 취하게 되면, 평균으로부터 변화(fluctuation)을 해석하는데도 용이하고, 평균으로 단순화된 또는 근사된 모델을 사용하게 되면 전체를 조망하기에 좋아진다. 

#### CRF 수식 
![](http://i.imgur.com/VHUFWCv.png)

unary term과 pairwise term으로 구성이 된다. 

- x : 각 픽셀의 위치에 해당하는 픽셀의 label
- i와 j는 픽셀의 위치
- Unary term은 CNN 연산을 통해서 얻을 수 있으며, 
- Pairwise term에서는 마치 bi-lateral filter에서 그러듯이 픽셀값의 유사도와 위치적인 유사도를 함께 고려한다.
    - 픽셀간의 detail한 예측에는 pairwise term이 중요한 역할을 한다. 

- 2개의 가우시안 커널로 구성이 된 것을 알 수 있으며, 표준편차 σα, σβ, σγ를 통해 scale을 조절할 수 있다
    - 첫번째 가우시안 커널은 비슷한 위치 비슷한 컬러를 갖는 픽셀들에 대하여 비슷한 label이 붙을 수 있도록 해주고, 
    - 두번째 가우시안 커널은 원래 픽셀의 근접도에 따라 smooth 수준을 결정한다. 
    - 위 식에서 pi, pj는 픽셀의 위치(position)를 나타내고, 
    - Ii, Ij는 픽셀의 컬러값(intensity)이다.
    
## 3. DeepLab V2 전체 동작 과정 

![](http://i.imgur.com/BpstsGs.png)

- DCNN을 통해 1/8 크기의 coarse score-map을 구하고, 
- 이것을 bi-linear interpolation을 통해 원영상 크기로 확대를 시킨다. 
- Bilinear interpolation을 통해 얻어진 결과는 각 픽셀 위치에서의 label에 대한 확률이 되며 이것은 CRF의 unary term에 해당이 된다. 
- 최종적으로 모든 픽셀 위치에서 pairwise term까지 고려한 CRF 후보정 작업을 해주면 최종적인 출력 결과를 얻을 수 있다.

--- 

<a name="1">[1]</a> [Semantic image segmentation with deep convolutional nets and fully connected CRFs](http://arxiv.org/pdf/1412.7062.pdf)  <br/>

<a name="2">[2]</a> [DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs](http://arxiv.org/pdf/1606.00915.pdf)  <br/>

