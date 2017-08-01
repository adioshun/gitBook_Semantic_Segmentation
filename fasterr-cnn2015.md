|논문명|Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks|
|-|-|
|저자(소속)|Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun (MS)|
|학회/년도|NIPS 2015, [논문](https://arxiv.org/pdf/1506.01497.pdf), [NIPS](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pd), [발표자료_ICCV15](http://kaiminghe.com/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf) |
|키워드|“attention” mechanisms,fully convolutional network |
|참고|[PR-012동영상(K)](https://youtu.be/kcPAGIgBGRs), [Ardias동영상(E)](https://www.youtube.com/watch?v=c1_g6tw69bU), [Jamie Kang(K)](https://jamiekang.github.io/2017/05/28/faster-r-cnn/), [Curt-Park(K)](https://curt-park.github.io/2017-03-17/faster-rcnn/), [Krzysztof Grajek(E)](https://softwaremill.com/counting-objects-with-faster-rcnn)|
|코드|[Caffe](https://github.com/rbgirshick/py-faster-rcnn), [PyTorch](https://github.com/longcw/faster_rcnn_pytorch), [TF](https://github.com/smallcorgi/Faster-RCNN_TF), [MatLab]( https://github.com/ShaoqingRen/faster_rcnn)|

# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

---
> [텐서플로우 블로그](https://tensorflow.blog/2017/06/05/from-r-cnn-to-mask-r-cnn/)

Fast R-CNN에서 남은 한가지 성능의 병목은 바운딩 박스를 만드는 리전 프로포잘 단계입니다. 

Faster R-CNN은 리전 프로포잘 단계를 CNN안에 넣어서 마지막 문제를 해결했습니다. 

CNN을 통과한 특성 맵에서 슬라이딩 윈도우를 이용해 각 지점anchor마다 가능한 바운딩 박스의 좌표와 이 바운딩 박스의 점수를 계산합니다. 

대부분 너무 홀쭉하거나 넓은 물체는 많지 않으므로 2:1, 1:1, 1:2 등의 몇가지 타입으로도 좋다고 합니다. 

Faster R-CNN은 작년에 마이크로소프트에서 내놓은 대표적인 컴퓨터 비전 연구 결과 중 하나입니다.

---
> [curt-park](https://curt-park.github.io/2017-03-17/faster-rcnn/)


## 1. Region Proposal Networks
입력 : image를 입력
출력 : 사각형 형태의 Object Proposal + Objectness Score
형태 : Fully convolutional network


### 1.1 Anchor Box
Anchor box는 sliding window의 각 위치에서 Bounding Box의 후보로 사용되는 상자

![](http://i.imgur.com/AeQXiE8.png)

동일한 크기의 sliding window를 이동시키며 window의 위치를 중심으로 사전에 정의된 다양한 비율/크기의 anchor box들을 적용하여 feature를 추출하는 것이다. 

장점 : 계산효율이 높은 방식이라 할 수 있다. 
- image/feature pyramids처럼 image 크기를 조정할 필요가 없으며, 
- multiple-scaled sliding window처럼 filter 크기를 변경할 필요도 없으므로 

## 2. Computation Process

### 2.1 입력 
- Shared CNN에서 convolutional feature map(14X14X512 for VGG)을 입력받는다. 
  - 여기서는 Shared CNN으로 VGG가 사용되었다고 가정한다. (Figure3는 ZF Net의 예시 - 256d)

> RPN은 sliding window에 3×3 convolution을 적용해 input feature map을 256 (ZF) 또는 512 (VGG) 크기의 feature로 mapping합니다.

### 2.2 Intermediate Layer
- 3X3 filter with 1 stride and 1 padding을 512개 적용하여 14X14X512의 아웃풋을 얻는다.

### 2.3 Output layer
> classification layer (cls)와 box regression layer (reg)으로 들어갑니다. Box classification layer와 box regression layer는 각각 1×1 convolution으로 구현됩니다.

> Box regression을 위한 초기 값으로 anchor라는 pre-defined reference box를 사용합니다. 이 논문에서는 3개의 크기와 3개의 aspect ratio를 가진 총 9개의 anchor를 각 sliding position마다 적용하고 있습니다.


#### A. cls layer
- 1X1 filter with 1 stride and 0 padding을 9*2(=18)개 적용하여 14X14X9X2의 이웃풋을 얻는다. 
- filter의 개수 : anchor box의 개수(9개) * score의 개수(2개: object? / non-object?)로 결정된다.

특정 anchor에 positive label이 할당되는 데에는 다음과 같은 기준이 있다.

1. 가장 높은 Intersection-over-Union(IoU)을 가지고 있는 anchor.
2. IoU > 0.7 을 만족하는 anchor.
  - IoU < 0.3 : non-positive anchor



#### B. reg layer
- 1X1 filter with 1 stride and 0 padding을 9*4(=36)개 적용하여 14X14X9X4의 아웃풋을 얻는다. 
- 여기서 filter의 개수는, anchor box의 개수(9개) * 각 box의 좌표 표시를 위한 데이터의 개수(4개: x, y, w, h)로 결정된다.


###### [참고] output layer에서 사용되는 파라미터의 개수 (VGG-16을 기준)
- 약 2.8 X 10^4개의 파라미터를 갖게 되는데(512 X (4+2) X 9), 
- 다른 모델(eg.GoogleNet:6.1 X 10^6-)보다 적다  
  - 이를 통해 small dataset에 대한 overfitting의 위험도가 상대적으로 낮으리라 예상할 수 있다.


## 3. Loss Function 
> 첫 논문에는 없다가 추후 [ICCV 2015 튜토리얼](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf)에서 추가 발표 

### 3.1 전체 Loss Function

![](http://i.imgur.com/NJnrGcR.png)
- pi: Predicted probability of anchor
- pi*: Ground-truth label (1: anchor is positive, 0: anchor is negative)
- lambda: Balancing parameter. Ncls와 Nreg 차이로 발생하는 불균형을 방지하기 위해 사용된다.
  -  cls에 대한 mini-batch의 크기가 256(=Ncls)이고, 이미지 내부에서 사용된 모든 anchor의 location이 약 2,400(=Nreg)라 하면 lamda 값은 10 정도로 설정한다.
- ti: Predicted Bounding box
- ti*: Ground-truth box

### 3.2 $$L_{reg}$$ 부분 
Bounding box regression 과정(Lreg)에서는 4개의 coordinate들에 대해 다음과 같은 연산을 취한 후,

![](http://i.imgur.com/yJRNbc2.png)

- x,y,w,h : 박스의 중앙 위치와 넓이, 높이 
- $$x$$ : 예측된 박스 
- $$x_a$$ : Achor 박스
- $$x^*$$ : Ground-truth 박스 

### 3.3 $$L_{loc}$$ 부분 
Smooth L1 loss function(아래)을 통해 Loss를 계산한다.

![](http://i.imgur.com/qupSFeb.png)

R-CNN / Fast R-CNN에서는 모든 Region of Interest가 그 크기와 비율에 상관없이 weight를 공유했던 것에 비해, 이 anchor 방식에서는 k개의 anchor에 상응하는 k개의 regressor를 갖게된다.

## 4. Training RPNs시 사용한 파라미터 
- end-to-end로 back-propagation 사용.
- Stochastic gradient descent
- 한 이미지당 랜덤하게 256개의 sample anchor들을 사용. 
  - 이때, Sample은 positive anchor:negative anchor = 1:1 비율로 섞는다. 
  - 혹시 positive anchor의 개수가 128개보다 낮을 경우, 빈 자리는 negative sample로 채운다. 
  - 이미지 내에 negative sample이 positive sample보다 훨씬 많으므로 이런 작업이 필요하다.
- 모든 weight는 랜덤하게 초기화.
  - from a zero-mean Gaussian distribution with standard deviation 0.01
- ImageNet classification으로 fine-tuning 
  - (ZF는 모든 layer들, VGG는 conv3_1포함 그 위의 layer들만. Fast R-CNN 논문 4.5절 참고.)
- Learning Rate: 0.001 (처음 60k의 mini-batches), 0.0001 (다음 20k의 mini-batches)
- Momentum: 0.9
- Weight decay: 0.0005

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

> convolutional feature map을 입력 받는다 . ZF Net의 예시 - 256d

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


```
4-step alternating training

1. Train RPNs
2. Train Fast R-CNN using the proposals from RPNs
3. Fix the shared convolutional layers and fine-tune unique layers to RPN
4. Fine-tune unique layers to Fast R-CNN
```

### B. 추후 제안 방법 ([ICCV 2015 튜토리얼](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf))

![](http://i.imgur.com/d7RwSeh.png)

RPN의 loss function과 Fast R-CNN의 loss function을 모두 합쳐 multi-task loss로 정의 하여 해결 
--

> [jamiekang](https://jamiekang.github.io/2017/05/28/faster-r-cnn/)

## 1. 관련 용어들 

### 1.1 [Hard Negative Mining](https://www.reddit.com/r/computervision/comments/2ggc5l/what_is_hard_negative_mining_and_how_is_it/)

Hard Negative Mining은 positive example과 negative example을 균형적으로 학습하기 위한 방법입니다.

단순히 random하게 뽑은 것이 아니라 confidence score가 가장 높은 순으로 뽑은 negative example을 (random하게 뽑은 positive example과 함께) training set에 넣어 training합니다.

![](http://i.imgur.com/jDCslgl.png)

### 1.2 Non Maximum Suppression
Non Maximum Suppression은 edge thinning 기법으로, 여러 box가 겹치게 되면 가장 확실한 것만 고르는 방법입니다. 

![](http://i.imgur.com/kbuSYIw.png)

### 1.3 Bounding Box Regression
Bound box의 parameter를 찾는 regression을 의미합니다. 

초기의 region proposal이 CNN이 예측한 결과와 맞지 않을 수 있기 때문입니다. 

Bounding box regressor는 CNN의 마지막 pooling layer에서 얻은 feature 정보를 사용해 region proposal의 regression을 계산합니다. 

