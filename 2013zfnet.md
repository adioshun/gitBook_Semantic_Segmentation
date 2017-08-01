# ZF Net ( =Feature Visualization )

> Visualizing Convolutional Neural Networks for Image Classification (2015, Danel Brukner)
> Delving Deep into Convolutional Nets (2014, Ken Chatfield)

> 자세한 내용(한글) : [Back-Projecting High-Level Activations](https://blog.lunit.io/2017/04/09/back-projecting-mid-level-activations/)

## 1. 개요 

ILSVRC 2013년 대회에서 우승을 한 구조로 뉴욕대의Matthew Zeiler와 Rob Fergus에 의해서 개발이 되었다.

Zeiler와 Fergus의 앞 글자를 따 ZF Net이라고 알려지게 되었다.

AlexNet의 hyper-parameter를 수정하여 성능을 좀 더 개선(3%)을 하였으며,특히 중간에 있는 convolutional layer의 크기를 늘렸다.

> ZFNet은 특정구조를 가리키는개념이 아니라, CNN을 보다 잘 이해할 수있는기법을 가리키는개념
> -  AlexNet의 기본 구조를 자신들의 Visualizing 기법을 통해 하이퍼파라미터 최적화 


## 2. 특징 : Deconvolution을 이용한 Visualization


Visualizing 기법 : 중간 layer에서 feature의 activity를 다시 입력 이미지 공간에 mapping을 시키는 기법

![](http://i.imgur.com/ep9d371.png)   

Deconvolution : 특정 feature의 activity가 입력 이미지에서 어떻게 mapping 되는지를 이해하기 위해 convolution(Feature map - filter - ReLU - Pooling) 과정을 역(reverse)으로 수행 하는것 
    - A. Pooling 
    - B. ReLU 
    - C. Filter 
    - D. Feature map 
    
###### A. Pooling 
                
- 문제점 : max-pooling 에 대한 역(reverse)를 구하는 것
    - 큰 값만 전달 하므로 `어떤 위치`에 있는 신호가 가장 강한 신호인지 파악할 수 있는 방법

- 해결책 : Switch 개념 활용 
    - switch는 가장 강한 자극의 위치 정보를 갖고 있는 일종의 꼬리표(flag)

![](http://i.imgur.com/4sN1FTB.png)

오른쪽 녹색화살표에서 작은 값은 왼쪽 빨간색 화살표에서 보면 사라져 있다. 
강한 값의 위치를 중간(Max Locations "Switches")에 저장하여 추후 복구시 사용 

###### B. ReLU 

음의 값을 갖는 것들은 0으로 정류(rectify) 하고 0이상 값들은 그대로 pass  시키기 때문에, 역과정을 수행할 때 양의 값들은 문제가 없고, 0 이하의 값들만 정류되었기 때문에 복원할 방법이 없다.

영향이 미미 하다고 하여 그냥 Skip 

###### C. Filter 

inverse filter 연산을 수행: 이 과정은 가역이 가능한 과정이기 때문에 특별히 문제가 될 것이 없다.


###### D. Feature map

시각화 해야 하는 대상 

eg. AlexNet (5개의 Conv 레이어) 

|Layer 1, 2|![](http://i.imgur.com/ZrE4ScP.png)|주로 영상의 코너나 edge 혹은 컬러와 같은 low level feature를 탐지|
|-|-|-|
|Layer 3|![](http://i.imgur.com/uiKqBgN.png)|비슷한 외양(texture)를 갖고 있는 특징을 추출|
|Layer 4,5|![](http://i.imgur.com/ygfdhAO.png)|Layer4에서는 사물이나 개체의 일부분<br>ayer5에서는 위치나 자세 변화|

이렇게 시각화(Visualizing)를 수행하게 되면,

- 특정 단계에서 얻어진 feature-map 들이 고르게 확보되어 있는지 혹은 특정 feature 쪽으로 쏠려 있는지,

- 개체의 pose나 기타 invariance 등을 잘 반영하는지 등등을 파악

학습의 결과에 대한 양호 여부를 판단할 수도 있고, 결과적으로 CNN의 구조가 적절했는지 등을 판단하기에 좋다.

---

## 3. 구조 

|![](http://i.imgur.com/kw4nAJl.png)|![](http://i.imgur.com/wQ4jGr0.png)|
|-|-|
|전체 구조 | AlexNet과 구조가 다른 첫번째와 두번째 layer|
ZFNet은 자신들의 Feature Visualization 기법의 효과를 보이는 것에 집중을 했기 때문에 전반적으로 AlexNet과 비슷한 구조를 취했다.
- AlexNet은 첫번째 convolution layer에 11x11 크기에 stride를 4로 정했는데,
- ZFNet은 filter 크기를 7x7로 하고, stride를 2로 하였다.
    - 그렇게 얻은 110x110 크기의 feature map에 대하여 3x3 overlapped max pooling을 수행하였으며, stride를 2로 하여 55x55 크기를 얻었다.

filter의 크기를 작게 하고 stride 크기를 작게 설정하도록 구조 변경하고, Feature 추출에 좋다는 것을 밝힘 

![](http://i.imgur.com/a85992r.png)
- AlexNet : b,d
- ZFNet : c, e




--- 
