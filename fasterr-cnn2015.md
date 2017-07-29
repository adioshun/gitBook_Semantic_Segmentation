|논문명|Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks|
|-|-|
|저자(소속)|Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun (MS)|
|학회/년도|NIPS 2015, [논문](https://arxiv.org/pdf/1506.01497.pdf), [NIPS](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pd) |
|키워드|“attention” mechanisms,fully convolutional network |
|참고|[PR-012동영상(K)](https://www.youtube.com/watch?v=kcPAGIgBGRs&feature=youtu.be&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS), [Ardias동영상(E)](https://www.youtube.com/watch?v=c1_g6tw69bU), [Jamie Kang(K)](https://jamiekang.github.io/2017/05/28/faster-r-cnn/), [Curt-Park(K)](https://curt-park.github.io/2017-03-17/faster-rcnn/), [Krzysztof Grajek(E)](https://softwaremill.com/counting-objects-with-faster-rcnn)|
|코드|[Caffe](https://github.com/rbgirshick/py-faster-rcnn), [PyTorch](https://github.com/longcw/faster_rcnn_pytorch), [MatLab]( https://github.com/ShaoqingRen/faster_rcnn)|

# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

---
> [텐서플로우 블로그](https://tensorflow.blog/2017/06/05/from-r-cnn-to-mask-r-cnn/)

Fast R-CNN에서 남은 한가지 성능의 병목은 바운딩 박스를 만드는 리전 프로포잘 단계입니다. 

Faster R-CNN은 리전 프로포잘 단계를 CNN안에 넣어서 마지막 문제를 해결했습니다. 

CNN을 통과한 특성 맵에서 슬라이딩 윈도우를 이용해 각 지점anchor마다 가능한 바운딩 박스의 좌표와 이 바운딩 박스의 점수를 계산합니다. 

대부분 너무 홀쭉하거나 넓은 물체는 많지 않으므로 2:1, 1:1, 1:2 등의 몇가지 타입으로도 좋다고 합니다. 

Faster R-CNN은 작년에 마이크로소프트에서 내놓은 대표적인 컴퓨터 비전 연구 결과 중 하나입니다.

---
> [라온피플 블로그](http://laonple.blog.me/220782324594)

## 1. 개요 

Fast R-CNN의 기본 구조와 비슷하지만, Region Proposal Network(RPN)이라고 불리는 특수한 망이 추가

RPN을 이용하여 object가 있을만한 영역에 대한 proposal을 구하고 그 결과를 RoI pooling layer에 보낸다. RoI pooling 이후 과정은 Fast R-CNN과 동일하다.

## 2. RPN

### 2.1  입력
- ConvNet 부분의 최종 feature-map

- 입력의 크기에 제한이 없음(Fast R-CNN에서 사용했던 동일한 ConvNet을 그대로 사용하기 때문)

### 2.2 동작 

- n x n 크기의 sliding window convolution을 수행하여 256 차원 혹은 512차원의 벡터(후보영역??)를 만들어내고

### 2.2 출력

- box classification (cls) layer :  물체인지 물체가 아닌지를 나타내는 
  - 출력 2k : object인지 혹은 object가 아닌지를 나타내는 2k score
  
- box regressor (reg) layer : 후보 영역의 좌표를 만들어 내는 에 연결한다.
  - 출력 4k : 4개의 좌표(X, Y, W, H) 값


> model의 형태 : Fully-convolutional network 형태

![](http://i.imgur.com/SH43wOr.png)
  
- 각각의 sliding window에서는 총 k개의 object 후보를 추천할 수 있으며, 

- 이것들은 sliding window의 중심을 기준으로 scale과 aspect ratio를 달리하는 조합(논문에서는 anchor라고 부름)이 가능하다. 
  - 논문에서는 scale 3가지와 aspect ratio 3가지를 지원하여, 총 9개의 조합이 가능하다.

- sliding window 방식을 사용하게 되면, anchor와 anchor에 대하여 proposal을 계산하는 함수가 “translation-invariant하게 된다. 
  - translation-invariant한 성질로 인해 model의 수가 크게 줄어들게 된다. 
  - k= 9인 경우에 (4 + 2) x 9 차원으로 차원이 줄어들게 되어, 결과적으로 연산량을 크게 절감

---


> [R-CNNs Tutorial](https://blog.lunit.io/2017/06/01/r-cnns-tutorial/)

## 1. RPN: Region Proposal Network

### 1.1 동작 과정 

![](http://i.imgur.com/Vd93ngo.png)

- Feature map 위의 N $$\times$$ N 크기의 작은 window 영역을 입력으로 받고,  
  - 하나의 feature map에서 모든 영역에 대해 물체의 존재 여부를 확인하기 위해서는 앞서 설계한 작은 N $$\times$$ N 영역을 sliding window 방식으로 탐색하면 될 것입니다. 
  

- 해당 영역에 물체가 존재하는지/존재하지 않는지에 대한 binary classification을 수행하는 작은 classification network를 만들어 볼 수 있습니다.  

- R-CNN, Fast R-CNN에서 사용되었던 bounding-box regression 또한 위치를 보정해주기 위해 추가로 사용됩니다. 

- 이러한 작동 방식은 N $$\times$$ N 크기의 convolution filter, 그리고 classification과 regression을 위한 1 $$\times$$ 1 convolution filter를 학습하는 것으로 간단하게 구현할 수 있습니다.

### 1.2 Anchor

문제점 : 다양한 후보영역 크기로 인해 고정된 N $$\times$$ N 크기의 입력만 처리 어려움 

미리 정의된 여러 크기와 비율의 reference box _k_를 정해놓고 각각의 sliding-window 위치마다 k개의 box를 출력하도록 설계하고 이러한 방식을 anchor라고 명칭
  - 하나의 feature map에 총 W $$\times$$ H $$\times$$ k개의 anchor가 존재하게 됩니다. 
  - eg. anchor k = 3가지의 크기(128, 256, 512), 3가지의 비율(2:1, 1:1, 1:2)   =9

RPN의 출력값은, 모든 anchor 위치에 대해 
- 각각 물체/배경을 판단하는 2k개의 classification 출력과, 
- x,y,w,h 위치 보정값을 위한 4k개의 regression 출력을 지니게 됩니다.

## 1.3 RPN 학습 과정

### A. Alternating optimization (임시, 논문에 기술된 방법)

RPN과 Fast R-CNN이 서로 convolution feature를 공유한 상태에서 번갈아가며 학습을 진행하는 복잡한 형태
- (NIPS 논문 제출로 인하여 급히 만든 임시 버젼이라고 밝힘)

ImageNet 데이터로 미리 학습된 CNN M0를 준비합니다.

- M0 conv feature map을 기반으로 RPN M1를 학습합니다.
- RPN M1을 사용하여 이미지들로부터 region proposal P1을 추출합니다.
- 추출된 region proposal P1을 사용해 M0를 기반으로 Fast R-CNN을 학습하여 모델 M2를 얻습니다.
- Fast R-CNN 모델 M2의 conv feature를 모두 고정시킨 상태에서 RPN을 학습해 RPN 모델 M3을 얻습니다.
- RPN 모델 M3을 사용하여 이미지들로부터 region proposal P2을 추출합니다.
- RPN 모델 M3의 conv feature를 고정시킨 상태에서 Fast R-CNN 모델 M4를 학습합니다.


### B. 추후 제안 방법 (ICCV 2015 튜토리얼)

![](http://i.imgur.com/d7RwSeh.png)

RPN의 loss function과 Fast R-CNN의 loss function을 모두 합쳐 multi-task loss로 정의 하여 해결 
