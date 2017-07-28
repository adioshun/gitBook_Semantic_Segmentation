|논문명|Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks|
|-|-|
|저자(소속)|Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun (MS)|
|학회/년도|NIPS 2015, [논문](https://arxiv.org/pdf/1506.01497.pdf), [NIPS](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pd) |
|키워드|“attention” mechanisms,fully convolutional network |
|참고|[PR-012동영상](https://www.youtube.com/watch?v=kcPAGIgBGRs&feature=youtu.be&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS), [Ardias논문리뷰](https://www.youtube.com/watch?v=c1_g6tw69bU), [코드](https://github.com/rbgirshick/py-faster-rcnn)|

# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

---
# 라온피플 블로그 

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

