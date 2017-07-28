|논문명/저자/소속|Fast R-CNN|
|-|-|
|저자(소속)|Ross Girshick(MS)|
|학회/년도|ICCV 2015, [논문](https://arxiv.org/abs/1504.08083)|
|키워드||
|참고|[코드_Python](https://github.com/rbgirshick/fast-rcnn), [다이어그램](https://drive.google.com/file/d/0B6Ry8c3OoOuqaWI3NGh2RERILVk/view?usp=sharing),[정리(한글)](http://judelee19.github.io/machine_learning/fast_rcnn/) [Caffe](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf)|

# Fast Region-based Convolutional Network method (Fast R-CNN)

--- 
> [텐서플로우 블로그](https://tensorflow.blog/2017/06/05/from-r-cnn-to-mask-r-cnn/)

R-CNN의 문제점은 모든 바운딩 박스마다 CNN을 돌려야 하고 분류를 위한 SVM, 바운딩 박스를 위한 선형 회귀까지 세가지 모델을 모두 훈련시키기 어렵다는 점입니다. Fast R-CNN은 이 문제들을 해결했습니다. 먼저 바운딩 박스들 사이에 겹치는 영역이 많은데 이들을 따로 따로 CNN을 통과시키는 것은 비용 낭비라고 생각했습니다. 여기에서 RoIPoolRegion of Interest Pooling의 개념을 도입하여 셀렉티브 서치에서 찾은 바운딩 박스 정보를 CNN을 통과하면서 유지시키고 최종 CNN 특성 맵으로 부터 해당 영역을 추출하여 풀링pooling합니다. 이렇게 하면 바운딩 박스마다 CNN을 돌리는 시간을 획기적으로 단축할 수 있습니다. 또한 SVM와 선형 회귀 모델을 모두 하나의 네트워크에 포함시켜 훈련을 시킵니다. SVM 대신 CNN 뒤에 소프트맥스softmax를 놓고, 선형 회귀 대신 소프트맥스 레이어와 동일하게 CNN에 뒤에 따로 추가했습니다.

---
> 라온피플 블로그 

|R-CNN|SPPNet|
|-|-|
|![](http://i.imgur.com/IASEVnA.png)|![](http://i.imgur.com/7FvA0FA.png)|

## 1. 개요 

### 1.1 R-CNN 문제점

- Training이 3 단계로 이루어짐.
  - 우선 약 2000여개의 후보 영역에 대하여 log loss 방식을 사용하여 fine tuning을 한다. 
  - 이후 ConvNet 특징을 이용하여 SVM에 대한 fitting 작업을 진행한다. 
  - 끝으로 bounding box regressor(검출된 객체의 영역을 알맞은 크기의 사각형 영역으로 표시하고 위치까지 파악)에 대한 학습을 한다.

- Training 시간이 길고 대용량 저장 공간이 필요.
  - VM과 bounding box regressor의 학습을 위해, 영상의 후보 영역으로부터 feature를 추출하고 그것을 디스크에 저장한다. 
  - eg. PASCAL VOC07 학습 데이터 5천장에 대하여 2.5일 정도가 걸리며, 저장 공간도 수백 GigaByte를 필요로 한다.



- 객체 검출(object detection) 속도가 느림.

  - 학습이 오래 걸리는 문제도 있지만, 
  - 실제 검출할 때도 875MHz로 오버클럭킹된 K40 GPU에서 영상 1장을 처리하는데 47초가 걸린다.

|문제점|학습시 모든 후보 영역(약 2,000개)에 대하여 개별적 연산을 실시 |
|-|-|
|해결책|Spatial Pyramid Pooling을 사용하여 convolution연산을 공유할 수 있는 방법|

### 1.2 patial Pyramid Pooling(SPPNet)의 문제점 

- Training이 3 단계로 이루어짐.

- Training 대용량 저장 공간이 필요.

- 정확도 문제 : Spatial Pyramid Pooling 앞단에 있는 convolutional layer에 대해서는 fine tuning 미실시

SPPNet은 ConvNet 단계는 전체 영상에 대하여 한꺼번에 연산을 하고 그 결과를 공유하고, SPP layer를 거치면서 region 단위 연산을 수행한다

## 2. Fast R-CNN

### 2.1 개요 

- 학습 시 multi-stage가 아니라 single-stage로 가능하고, 
  - Softmax / BBox Regressor를 병렬적으로 처리 

- 학습의 결과를 망에 있는 모든 layer에 update할 수 있어야 하며, 
  - 전체 영상에 대해 ConvNet 연산을 1번만 수행후 결과 공유 


- feature caching을 위해 별도의 디스크 공간이 필요 없는 방법

### 2.2 구조 

![](http://i.imgur.com/QSbwE7W.png)

- 전체 이미지 및 객체 후보 영역을 한꺼번에 받아들인다. 
  - Convolution과 max-pooling을 통해 이미지 전체를 한번에 처리를 하고 feature map을 생성한다.

- 각 객체 후보 영역에 대하여 `RoI Pooling layer`를 통해, feature-map으로부터 fixed-length feature 벡터를 추출한다.
  - 이 부분은 SPPNet의 Spatial Pyramid Pooling과 하는 일이 유사하다고 볼 수 있다.
  - RoI Pooling layer에서 다양한 후보 영역들에 대하여 FC layer로 들어갈 수 있도록 크기를 조정

- 추출된 fixed-length feature vector는 Fully-Connected Layer에 인가를 하며, 뒷단 2개의 모듈에 전달 
  - softmax : “object class + background”를 추정
  - bbox(bounding box) regressor : 각각의 object class의 위치를 출력
  
### 2.3 특징 (RoI Pooling layer)
> Fast R-CNN의 RoI Pooling layer와 SPP layer차이점 

- SPPNet에서 제안한 SPP layer는 feature map 상의 특정 영역에 대해 일정한 고정된 개수의 bin으로 영역을 나눈 뒤, 각 bin에 대해 max pooling 또는 average pooling을 취함으로써 고정된 길이의 feature vector를 가져올 수 있습니다. 

- Fast R-CNN에서는 이러한 SPP layer의 single level pyramid만을 사용하며, 이를 RoI Pooling layer라고 명칭하였습니다.

![](http://i.imgur.com/idoUX2g.png)


## 3. 학습과 테스트 

|학습|테스트|
|-|-|
|![](http://i.imgur.com/dGLAPVd.png)|![](http://i.imgur.com/FF7ais8.png)|

## 4. 샘플링 방법 
R-CNN과 SPPNet : region-wise sampling

Fast R-CNN : hierarchical sampling

> 상세내용은 [라온피플 블로그](http://laonple.blog.me/220752877630) 참고



---
