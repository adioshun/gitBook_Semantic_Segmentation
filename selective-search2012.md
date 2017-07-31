|논문명|Selective Search for Object Recognition|
|-|-|
|저자(소속)|J.R.R. Uijlings|
|학회/년도|IJCV 2012, [논문](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)|
|키워드||
|참고|[발표자료#1](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/ssearch_schuyler.pdf),[발표자료#2](http://www.cs.cornell.edu/courses/cs7670/2014sp/slides/VisionSeminar14.pdf)|

# Selective Search for Object Recognition
---
여러 후보 영역 추천 알고리즘 

![](http://i.imgur.com/W3sVgEb.png)
> 2015년 Jan Hosang 등이 발표한 논문 “What makes for effective detection proposals?

1등 : EdgeBox 
상위권 : Selective Search 




## 2. Selective Search 

### 2.1 개요 

객체 인식시 후보 영역 추천에 사용되는 알고리즘 (eg. R-CNN, SPPNet, Fast R-CNN + Selective Search)
- 인식과 추천이 두개의 서로 다른 알고리즘으로 분리되어 있어 End-to-End 학습은 어려움 
- 최근에는 인식 알고리즘이 추천 알고리즘도 포함하고 있음(eg. Faster R-CNN, YOLO, FCN)

SS의 목표
- 영상은 계층적 구조를 가지므로 적절한 알고리즘을 사용하여, 크기에 상관없이 대상을 찾아낸다.

- 컬러, 무늬, 명암 등 다양한 그룹화 기준을 고려한다.

- 빨라야 한다.

### 2.2 SS특징
exhaustive search 방식 + segmentation 방식 결합
- Exhaustive search : 후보가 될만한 모든 영역을 샅샅이 조사, scale, aspect ratio을 모두 탐색,계산 부하가 크다 
- segmentation : 영상 데이터의 특성(eg. 색상, 모양, 무늬)에 기반, 모든 경우에 동일하게 적용할 수 있는 특성 찾기 어려움

“bottom-up” 그룹화 방법 사용
- 영상에 존재한 객체의 크기가 작은 것부터 큰 것까지 모두 포함이 되도록 작은 영역부터 큰 영역까지 계층 구조를 파악
- 작은 Seed정보에서 확장하면서 점점 큰 Object화 시키는 것 

### 2.3 동작 원리 

논문에서는 이것을 segmentation 방법을 가이드로 사용한 data-driven SS라고 부른다.

1. segmentation에 동원이 가능한 다양한 모든 방법을 활용하여 seed를 설정하고, 
2. 그 seed에 대하여 exhaustive한 방식으로 찾는 것을 목표로 하고 있다. 

###### Step 1.  일단 초기 sub-segmentation을 수행한다.

이 과정에서는 Felzenszwalb가 2004년에 발표한 “Efficient graph-based image segmentation” 논문의 방식처럼, 각각의 객체가 1개의 영역에 할당이 될 수 있도록 많은 초기 영역을 생성하며, 아래 그림과 같다.

![](http://i.imgur.com/nrRCavf.png)

###### Step 2. 작은 영역을 반복적으로 큰 영역으로 통합한다.

이 때는 “탐욕(Greedy) 알고리즘을 사용하며, 그 방법은 다음과 같다. 우선 여러 영역으로부터 가장 비슷한 영역을 고르고, 이것들을 좀 더 큰 영역으로 통합을 하며, 이 과정을 1개의 영역이 남을 때까지 반복을 한다.

아래 그림은 그 예를 보여주며, 초기에 작고 복잡했던 영역들이 유사도에 따라 점점 통합이 되는 것을 확인할 수 있다.

![](http://i.imgur.com/SKVJ3C1.png)


###### Step 3. 통합된 영역들을 바탕으로 후보 영역을 만들어 낸다.

![](http://i.imgur.com/RHvrATC.png)

[정리]
- 입력 영상에 대하여 segmentation을 실시하면, 이것을 기반으로 후보 영역을 찾기 위한 seed를 설정한다.
- 엄청나게 많은 후보가 만들어지게 되며, 이것을 적절한 기법을 통하여 통합을 해나가면,
- segmentation은 [3]형태로 바뀌게 되며, 결과적으로 그것을 바탕으로 후보 영역이 통합되면서 개수가 줄어들게 된다.



#### A. sub-segmentation 알고리즘 (Step 1상세설명)

> 2.3 동작원리의 Step 1에 필요한 알고리즘 

“Efficient Graph-based Image Segmentation”, Felzenszwalb, 2004

##### 가.  개요 

논문에서는 사람이 인지하는 방식으로의 segmentation을 위해 graph 방식을 사용하였다.  

그래프 이론 G = (V, E)에서 V는 노드(virtex)를 나타내는데, 여기서는 픽셀이 바로 노드가 된다.

기본적으로 이 방식에서는 픽셀들 간의 위치에 기반하여 가중치(w)를 정하기 때문에 “grid graph 가중치” 방식이라고 부르며, 가중치는 아래와 같은 수식으로 결정이 되고 graph는 상하좌우 연결된 픽셀에 대하여 만든다.

![](http://i.imgur.com/dWKd13n.png)

- E(edge)는 픽셀과 픽셀의 관계를 나타내며 가중치 w(vi, vj)로 표현이 되는데, 
    - 가중치(w)는 픽셀간의 유사도가 떨어질수록 큰 값을 갖게 되며, 
    - 결과적으로 w 값이 커지게 되면 영역의 분리가 일어나게 된다.

##### 나.  분리 or 통합 판단 수식 

![](http://i.imgur.com/o01Q0ko.png)
![](http://i.imgur.com/zAar8kO.png)
- Dif(C1, C2)는 두개의 그룹을 연결하는 변의 최소 가중치를 나타내고, 
- MInt(C1, C2)는 C1과 C2 그룹에서 최대 가중치 중 작은 것을 선택한 것이다. 

즉, 그룹간의 차가 그룹 내의 차보다 큰 경우는 별개의 그룹으로 그대로 있고, 그렇지 않은 경우에는 병합을 하는 방식이다.

##### 다.  Nearest Neighbor graph 가중치 

인접한 픽셀끼리, 즉 공간적 위치 관계를 따지는 방법뿐만 아니라 feature space에서의 인접도를 고려한 방식

적정한 연산 시간을 유지하기 위해 feature space에서 가장 가까운 10개의 픽셀에 한하여 graph를 형성한다.
- 모든 픽셀을 (x, y, r, g, b)로 정의된 feature space로 투영 : (x, y)는 픽셀의 좌표, (r, g, b)는 픽셀의 컬러 값
- 가중치에 대한 설정은 5개 성분에 대한 Euclidean distance를 사용하였다. 
- 그룹화 방식은 동일하다.

#### B. Hierarchical Grouping (Step 2상세설명)

1단계 sub-segmentation에서 만들어진 여러 영역들을 합치는 과정을 거쳐야 한다. 

SS에서는 여기에 **유사도(similarity metric)**를 기반으로 한 **greedy 알고리즘**을 적용하였다


![](http://i.imgur.com/3QktetY.png)

1. 먼저 segmentation을 통하여 초기 영역을 설정해준다. 
2. 그 후 인접하는 모든 영역들 간의 유사도를 구한다. 
3. 전체 영상에 대하여 유사도를 구한 후, 가장 높은 유사도를 갖는 2개의 영역을 병합시키고, 병합된 영역에 대하여 다시 유사도를 구한다. 
4. 새롭게 구해진 영역은 영역 list에 추가를 한다. 
5. 이 과정을 전체 영역이 1개의 영역으로 통합될 때까지 반복적으로 수행을 하고, 
6. 최종적으로 R-리스트에 들어 있는 영역들에 대한 bounding box를 구하면 된다.

> [13] : Efficient Graph Based Image Segmentation.

### 2.4 성능향상을 위한 다양화 전략 (Diversification Strategy)

후보 영역 추천의 성능 향상을 위해 SS에서는 다음과 같은 다양화 전략을 사용한다.
- 다양한 컬러 공간을 사용 
- color, texture, size, fill 등 4가지 척도를 적용하여 유사도를 구하기

#### A. 다양한 컬러 공간

RGB뿐만 아니라, HSV등 8개의 Color공간 사용

![](http://i.imgur.com/gRF2QZE.png)
[각가의 컬러 공간이 영향을 받는 수준 ]

#### B. 다양한 유사도 검사의 척도

SS는 유사도 검사의 척도로 color, texture, size, fill을 사용을 하며, 유사도 결과는 모두 [0, 1] 사이의 값을 갖도록 정규화(normalization) 시킨다.

$$
유사도 = s(r_i, r_j) = a_1S_{colour}(r_i,r_j) + a_2S_{texture}(r_i,r_j) + a_3S_{size}(r_i, r_j) + a_4S_{fill}(r_i,r_j)

$$

##### 가. 컬러 유사도

- 컬러 유사도 검사에는 히스토그램을 사용한다. 
- 각각의 컬러 채널에 대하여 bin을 25로 하며, 히스토그램은 정규화 시킨다. 
- 각각의 영역들에 대한 모든 히스토그램을 구한 후 인접한 영역의 히스토그램의 교집합 구하는 방식으로 유사도를 구하며, 식은 아래와 같다.

$$
S_{colour}(r_i,r_j) = \sum^n_{k=1}min(c^k_i, c^k_j)
$$


##### 나. texture 유사도

- SIFT(Scale Invariant Feature Transform)와 비슷한 방식을 사용하여 히스토그램을 구한다. 
- 8방향의 가우시안 미분값을 구하고 그것을 bin을 10으로 하여 히스토그램을 만든다. 
- SIFT는 128차원의 디스크립터 벡터를 사용하지만, 여기서는 80차원의 디스크립터 벡터를 사용하며, 컬러의 경우는 3개의 채널이 있기 때문에 총 240차원의 벡터가 만들어진다.

$$
S_{texture}(r_i,r_j) = \sum^n_{k=1}min(t^k_i, t^k_j)
$$

##### 다. 크기 유사도

- 작은 영역들을 합쳐서 큰 영역을 만들 때, 다른 유사도만 따지만 1개 영역이 다른 영역들을 차례로 병합을 하면서 영역들의 크기 차이가 나게 된다. 

- 크기 유사도는 작은 영역부터 먼저 합병이 되도록 해주는 역할을 한다. 일종의 가이드 역할을 하게 되는 것이다.

$$
S_{size}(r_i, r_j)= 1- \frac{size(r_i)+size(r_j)}{size(im)}
$$

- size(im)은 전체 영역에 있는 픽셀의 수이고, 
- size(ri)와 size(rj)는 유사도를 따지는 영역의 크기(픽셀 수)이다. 

영역의 크기가 작을수록 유사도가 높게 나오기 때문에, 다른 유사도가 모두 비슷한 수준이라면 크기가 작은 비슷한 영역부터 먼저 합병이 된다.


##### 라. fill 유사도

- 2개의 영역을 결합할 때 얼마나 잘 결합이 되는지를 나타낸다. 

- fill이라는 용어가 붙은 이유는 2개의 영역을 합칠 때 **gap이 작아지는** 방향으로 유도를 하기 위함이다. 

- 가령 1개의 영역이 다른 영역에 완전히 포함이 되어 있는 형태라고 한다면, 그것부터 합병을 시켜야 hole이 만들어지는 것을 피할 수 있게 된다.

![](http://i.imgur.com/TCTAwxY.png)

이 그림에서 r1과 r2를 합쳤을 경우 r1과 r2를 합친 빨간색의 bounding box 영역(BBij)의 크기에서 r1과 r2 영역의 크기를 뺐을 때 작은 값이 나올수록 2 영역의 결합성(fit)이 좋아지게 된다.

fill 유사도는 이런 방향으로 합병을 유도하기 위한 척도라고 보면 된다. 

$$
S_{fill}(r_i,r_j) = 1- \frac{size(BB_{ij})-size(r_i)-size(r_i)}{size(im)}
$$

Fit이 좋을수록 결과적으로 1에 근접한 값을 얻을 수 있기 때문에 그 방향으로 합병이 일어나게 된다.

### 2.5 SS 의 탐지 알고리즘 

> SS는 후보영역 추천 + 탐지 알고리즘으로 이루어져 있지만, 후보영역 추천만 많이 활용되고 있다. 탐지 알고리즘을 살펴 보려면 ["Selective Search(Part4)"](http://laonple.blog.me/220918802749)참고 

object detection을 위해 사용한 기본적인 접근법은 아래와 같다.

- SIFT 기반의 feature descriptor를 사용한 BoW(Bag of words) 모델

- 4레벨의 spatial pyramid

- 분류기로 SVM을 사용

#### A. Bag of Words 모델

![](http://i.imgur.com/fxs544P.png)

- 영상을 구성하는 특정한 성분(특징=Feature)들의 분포를 이용하여 그림의 오른쪽처럼 히스토그램으로  나타낼 수가 있게 된다. 
    - 확연히 다른 분포를 보인다면, 성분들의 분포만을 이용해 해당 object 여부를 판별할 수 있게 되는 것이다.
    - 이런 방식을 BoW(Bag of Words)라고 부른다.     

---
