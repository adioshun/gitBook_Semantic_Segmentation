|논문명 | |
| --- | --- |
| 저자\(소속\) | \(\) |
| 학회/년도 | IROS 2015, [논문]() |
| 키워드 | |
| 데이터셋(센서)/모델 | |
| 관련연구||
| 참고 | |
| 코드 | |








# SPPNet




> 참고 : [라온피플 블로그](http://laonple.blog.me/220692793375)

## 1. 개요 

- SPPNet은 마이크로소프트의 북경연구소에 근무하는 Kaiming He 등에 의해서 개발

- R-CNN의 문제점 개선에 초점 

> Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition

### 1.1 R-CNN의 문제점

- 이미지 왜곡 
    - AlexNet사용위해 입력크기를 224x224 변화위해 warping이나 crop을 사용
    - 이로 인한 이미지 변형이나 crop으로 인한 손실로 인해, 성능 저하가 일어날 수 있는 요인이 존재.

 

- 연산비용 큼 
    - 2000여개에 이르는 region proposal에 대해 순차적으로 CNN을 수행해야 하기 때문에 학습이나 실제 run time이 긴 문제.


- GPU 부적합 
    - 사용하는 알고리즘이 특히, region proposal이나, SVM 튜닝 등이 GPU 사용에 적합하지 않다는 점.
    
    
### 1.2 해결책

#### A. 이미지 왜곡

 ![](http://i.imgur.com/DgqNVr2.png) 
 - R-CNN : AlexNet입력을 위해 224x224 크기로 이미지를 Crop/warp 하여 사용 
 - SPPNet : pyramid 연산을 통해 입력 영상의 크기를 대응할 수 있게 되면, 굳이 crop/warp를 사용하지 않아도 된다.
 
대부분의 넷이 입력크기의 영향을 받는 이유는 fully-connected layer가 입력 영상의 크기에 제한을 받기 때문이다. 

여러 단계의 피라미드 레벨에서 오는 자잘한 feature들을 fully-connected layer의 입력으로 사용(BoW 개념 활용)
- 이때, 피라미드의 출력을 영상의 크기에 관계없이 사전에 미리 정하면 더 이상 영상의 크기에 제한을 받지 않게 됨

>  BoW(Bag-of-Words) : 특정 개체를 분류하는데 굵고 강한 특징에 의존하는 대신에서 작은 여러 개의 특징을 사용하면 개체를 잘 구별할 수 있다

#### B. 연산비용 큼 

- R-CNN : 각각의 후보 window에 대해 crop/warp를 한 후 CNN 과정을 전부 거치지만
- SPPNet : SPPNet에서는 영상 크기에 영향을 받지 않기 때문에 전체 영상에 대해 딱 1번 convolutional layer를 거친 후 해당 window에 대하여 SPP를 수행 (24 ~ 102 배 정도 빠르다)

## 2. 구조 

![](http://i.imgur.com/xZde5hv.png)

AlexNet의 5번째 convolutional layer 다음에 SPP layer가 위치를 하며, 이후에 fully connected layer가 오는 구조를 취한다.

## 3. 특징 (Spatial Pyramid Pooling, RoI Pooling) 

![](http://i.imgur.com/IPbiLQ3.png)

- Bag-of-words (BoW) : 다양한 크기의 입력으로부터 일정한 크기의 feature를 추출해 낼 수 있는 방법

- BoW는 이미지가 지닌 특징들의 위치 정보를 모두 잃어버린다는 단점이 존재

- 이미지를 여러개의 일정 개수의 지역으로 나눈 뒤, 각 지역에 BoW를 적용하여 지역적인 정보를 어느정도 유지


