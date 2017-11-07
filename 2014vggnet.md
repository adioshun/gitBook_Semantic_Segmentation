|논문명 | |
| --- | --- |
| 저자\(소속\) | \(\) |
| 학회/년도 | IROS 2015, [논문]() |
| 키워드 | |
| 데이터셋(센서)/모델 | |
| 관련연구||
| 참고 | |
| 코드 | |





# VGGNet

> ILSVRC 2014년 대회에서 2등, 
> Karen Simonyan,  Andrew Zisserman, "Very deep convolutional networks for large-scale image recognition", 영국 옥스포트 대학교

- CNN 연구 그룹에서는 VGGNet(ILSVRC 2014 2등)의 구조를 좀 더 선호하는 경향이 있다.
 - 장점 : GoogLeNet(ILSVRC 2014 1등)에 비해 분류 성능은 약간 떨어졌지만, 다중 전달 학습 과제에서는 오히려 더 좋은 결과가 나왔다, 간단한 구조이고, 쉽게 변형이 가능하여 많이 사용되고 있음 
 - 단점 : 메모리 수와 파라미터의 수가 크다는 점이다.


 
## 1. 개요 

- 목적 : 대량의 이미지를 인식함에 있어, 망의 깊이(depth)가 정확도에 어떤 영향을 주는지 실험 
 
 - 실험 환경 : receptive field의 크기는 가장 간단한 3x3으로 정하고 깊이가 어떤 영향을 주는지 6개의 구조(하단 표 A~E)에 대하여 실험
 
 - 실험 결과 : 망의 시작부터 끝까지 동일하게 3x3 convolution과 2x2 max pooling을 사용하는 단순한(homogeneous) 구조에서 depth가 16일 때 최적의 결과가 나오는 것을 보여줬다.

![](http://i.imgur.com/nHiQVNs.png)


## 2. 구조 
- 기본 구조 : Conv + Pooling 

- VGGNet 구조 : Conv + Conv + Conv + Pooling (3x3인 filter를  여러 개를 stack 시킴)
 - 3x3 convolutional layer를 2개 쌓으면 5x5 convolution이 되고, 3개를 쌓으면 7x7 convolution이 된다

![](http://i.imgur.com/bVpiBfz.png)

처음 4 layer와 마지막 fully-connected layer의 경우는 vanishing문제 해결을 위해 A의 학습 결과로 초기값을 설정 후 변경을 하지 않음 


## 3. 특징 / Insight

- VGGNet에서 A-LRN은 Local Response Normalization이 적용된 구조인데, 예상과 달리 VGGNet 구조에서는 LRN이 별 효과가 없어 사용하지 않는다.

- 1x1이 적용되기는 하지만, 적용하는 목적이 GoogLeNet이나 NIN의 경우처럼 차원을 줄이기 위한 목적이라기 보다는 차원은 그대로 유지하면서 ReLU를 이용하여 추가적인 non-linearity를 확보하기 위함이다.

- vanishing/exploding gradient 문제 : 11-layer의 학습 결과를 더 깊은 layer의 파라미터 초기화 시에 이용

## 4. VGGNet에서 성능을 끌어올리기 위해 적용한 방법

구조는 단순화 하고, Training / Testing 방법의 향상을 통해 성능 개선 

### 4.1 Training 방법

#### A. Data augmentation 

입력이미지의 Scale 조절 

- single-scale training 모드 : S = 256 OR S = 384 고정 

- multi-scaling training(=scale jittering) 모드 : $$S_{min}$$과 $$S_{max}$$ 범위에서 무작위로 선택
 

#### B. RGB 컬러 성분 변화 

### 4.2 Testing 방법

테스트 영상을 multi-crop 방식과 dense evaluation 개념을 섞어 사용


