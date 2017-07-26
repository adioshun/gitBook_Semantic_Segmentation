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
## 1. Abstract
This paper addresses the problem of generating possible object locations for use in object recognition. We introduce Selective Search which combines the strength of both an exhaustive search and segmentation.
- Like segmentation, we use the image structure to guide our sampling process. 
- Like exhaustive search, we aim to capture all possible object locations. 

> 기존 방법의 문제점 언급하고 Selective Search 알고리즘 제안. 제안 알고리즘은 기존 exhaustive search와 segmentation의 장점을 모두 수용하고 있음 
> - segmentation : 이미지 구조 활용
> - exhaustive search : 가능한 영역을 모두 Capture함 

Instead of a single technique to generate possible object locations, we diversify our search and use a variety of complementary image partitionings to deal with as many image conditions as possible. 

Our Selective Search results in a small set of data-driven, class-independent, high quality locations,yielding 99% recall and a Mean Average Best Overlap of 0.879 at 10,097 locations. 

The reduced number of locations compared to an exhaustive search enables the use of stronger machine learning techniques and stronger appearance models for object recognition.

In this paper we show that our selective search enables the use of the powerful Bag-of-Words model for recognition.

> 제안 방식은 Bag-of-Words model을 차용하였다. 
 
The Selective Search software is made publicly available. 



## 1 Introduction

For a long time, objects were sought to be delineated before their identification. 

> 물체는 윤곽을 찾고 이후 구분 작업이 후행 되었다. 

This gave rise to segmentation, which aims for a unique partitioning of the image through a generic algorithm, where there is one part for all object silhouettes in the image. 

> 따라서, segmentation기술이 중요하다. segmentation은 물체의 일부만 있어도 구분해 내는 기술이다. 

Research on this topic has yielded tremendous progress over the past years [3, 6, 13, 26]. 

> 이분야는 최근 많은 발전이 있었다. 

![](http://i.imgur.com/CTQY4II.png)

But images are intrinsically(본질적으로) hierarchical: In Figure 1a the salad and spoons are inside the salad bowl, which in turn stands on the table. Furthermore, depending on the context the term table in this picture can refer to only the wood or include everything on the table. 

> 사진은 본질적으로 계층적이다. 그림 1a를 보면, 테이블 위에 Bowl이 있고, 그 안에 Salad와 스푼이 있다. 

Therefore both the nature of images and the different uses of an object category are hierarchical. This prohibits the unique partitioning of objects for all but the most specific purposes. 

Hence for most tasks multiple scales in a segmentation are a necessity. 

This is most naturally addressed by using a hierarchical partitioning, as done for example by Arbelaez et al.[3].

> 이러한 이유로 hierarchical partitioning이 필요 하다. 

Besides that a segmentation should be hierarchical, a generic solution for segmentation using a single strategy may not exist at all.

There are many conflicting reasons why a region should be grouped together: In Figure 1b the cats can be separated using colour, but their texture is the same. Conversely, in Figure 1c the chameleon is similar to its surrounding leaves in terms of colour, yet its texture differs. 

> 여러 region이 하나의 그룹으로 처리 되는지에는 복잡한 문제가 있다. 그림 1b에서 고양이는 색으로는 구분 할수 있지만, texture 은 같다. 반대로, 그림 1c에서 카멜론은 색으로 구분이 어렵지만 texture 로 구분 할수 있다. 

Finally, in Figure 1d, the wheels are wildly different from the car in terms of both colour and texture, yet are enclosed by the car. Individual visual features therefore cannot resolve the ambiguity of segmentation.

> 결과적으로 그림 1d에서 바퀴와 차는 색과 질감으로 보자면 하나로 보기 어렵다. 즉, 하나식 물체를 보게 되면 ambiguity of segmentation문제를 풀기 어렵다. 

And, finally, there is a more fundamental problem. Regions with very different characteristics, such as a face over a sweater, can only be combined into one object after it has been established that the object at hand(가까이) is a human. Hence without prior recognition it is hard to decide that a face and a sweater are part of one object [29].

> 더 근본적인 문제는 스웨터 입은 사람을 구분하는 것처럼 매우 다른 특성을 가지고 있을때이다. 그러므로 사전적 인지 없이는 스웨터와 얼굴이 하나의 물체로 인지 하는것은 어렵다. 

This has led to the opposite of the traditional approach: to do localisation through the identification of an object. 

> 이러한 접근이 기존 방식을 반대로 이끌었다. : identification -> Localisation 

This recent approach in object recognition has made enormous progress in less than a decade [8, 12, 16, 35]. 

> 최근 몇년간 이러한 접근[8, 12, 16, 35]이 많은 발전을 이루었다. 

With an appearance model learned from examples, an exhaustive search is performed where every location within the image is examined as to not miss any potential object location [8, 12, 16, 35].However, the exhaustive search itself has several drawbacks. Searching every possible location is computationally infeasible.

> exhaustive search는 이미지안의 모든 영역을 살펴 봄으로써 실수 확률을 낮춘다. 하지만 모든 영역을 찾는건 현실적으로 불가능 하다. (컴퓨팅 파워) 

The search space has to be reduced by using a regular grid, fixed scales, and fixed aspect ratios. In most cases the number of locations to visit remains huge, so much that alternative restrictions need to be imposed. 

> 탐색 영역을 줄이기 위해 다음 기술이 사용 될수 있다. : regular grid, fixed scales, and fixed aspect ratios. 그래도 탐색 부분이 많기 떄문에 추가적인 기술이 필요 하다. 

The classifier is simplified and the appearance model needs to be fast. Furthermore, a uniform sampling yields many boxes for which it is immediately clear that they are not supportive of an object. Rather then sampling locations blindly using an exhaustive search, a key question is: Can we steer the sampling by a data-driven analysis?

> 여러 문제가 있다. 무보다 Can we steer the sampling by a data-driven analysis?인지 자문하게 된다. 

In this paper, we aim to combine the best of the intuitions of segmentation and exhaustive search and propose a data-driven selective search. 
- Inspired by bottom-up segmentation, we aim to exploit the structure of the image to generate object locations. 
- Inspired by exhaustive search, we aim to capture all possible object locations.

> 본 논문에서는 segmentation and exhaustive search를 합쳐서 Selective Search를 제안한다. 
> - segmentation에서는 이미지의 구조 탐색 후에 물체 위치를 찾는 방법을 채택하고 
> - exhaustive search에서눈 모든 가능한 영역을 모두 capture하는 방법을 채택했다. 

Therefore, instead of using a single sampling technique, we aim to diversify the sampling techniques to account for as many image conditions as possible. 

> 하나의 방법을 쓰는 샘플링 방법보다. 여러 이미지 상태에 따라 대처 가능한 샘플링 방법을 개발하였다. 

Specifically, we use a data-driven grouping based strategy where we increase diversity by using a variety of complementary grouping criteria and a variety of complementary colour spaces with different invariance properties. 

> 특히 data-driven grouping 기반 방식을 사용함으로써 다양성을 증대 시켰다. 
> - 다양한 complementary grouping criteria
> - 다양한 complementary colour spaces

The set of locations is obtained by combining the locations of these complementary partitionings. 

> 위치 정보는 이러한 complementary partitionings의 위치 정보를 합쳐서 획득 하였다. 

Our goal is to generate a class-independent,data-driven, selective search strategy that generates a small set of high-quality object locations.

Our application domain of selective search is object recognition. We therefore evaluate on the most commonly used dataset for this purpose, the Pascal VOC detection challenge which consists of 20 object classes. 

> 제안 알고리즘의 목적이 object recognition이므로 Pascal VOC을 이용하여 테스트 하였다. 


The size of this dataset yields computational constraints for our selective search. Furthermore, the use of this dataset means that the quality of locations is mainly evaluated in terms of bounding boxes. However, our selective search applies to regions as well and is also applicable to concepts such as “grass”.

> Pascal VOC 데이터셋은 계산상 제약을 가지고 있다. 또한, 평가시 BBox로 평가 방법을 제공한다. ????

## 2. Related Work
We confine the related work to the domain of object recognition and divide it into three categories: 
- Exhaustive search, 
- segmentation,
- and other sampling strategies that do not fall in either category

> object recognition관련된 부분만 크게 3부분으로 나누어 다룬다. 

### 2.1 Exhaustive Search
As an object can be located at any position and scale in the image, it is natural to search everywhere [8, 16, 36]. However, the visual search space is huge, making an exhaustive search computationally expensive. This imposes constraints on the evaluation cost per location and/or the number of locations considered.

> 이미지상에 물체는 다양한 크기로 어느 곳에나 위치 할수 있으므로 모든것을 찾아 보는것이 일반적인 방법이다. 그러나 이 방법은 검색공간이 커서 계산 부하가 크다. 

Hence most of these sliding window techniques use a coarse search grid and fixed aspect ratios, using weak classifiers and economic image features such as HOG [8, 16, 36]. 

> 그래서 이러한 슬라이딩 윈도우 기술은 coarse search grid와 fixed aspect ratios을 사용하고, 약한 식별자와 계산 효율이 적은 Feature(=HoG)를 하용한다. 

This method is often used as a pre-selection step in a cascade of classifiers [16, 36].

> 이 방법은 간혹 cascade of classifiers의 사전선택 스텝으로 사용되기도 한다. 

Related to the sliding window technique is the highly successful part-based object localisation method of Felzenszwalb et al. [12].Their method also performs an exhaustive search using a linear SVM and HOG features. However, they search for objects and object parts, whose combination results in an impressive object detection performance. Lampert et al. [17] proposed using the appearance model to guide the search. This both alleviates the constraints of using a regular grid, fixed scales, and fixed aspect ratio, while at the same time reduces the number of locations visited. This is done by directly searching for the optimal window within the image using a branch and bound technique. While they obtain impressive results for linear classifiers, [1] found that for non-linear classifiers the method in practice still visits over a 100,000 windows per image.

> sliding window관련 기술로는 [12]가 좋다. [12]도 Linear SMV/HOG을 이용해서 exhaustive search를 수행한다. 성능도 좋다. [17]은 탐색시 가이드로 appearance model를 이용한다. 위 두 방식은 기존 방식대비 제약을 줄여 주고, 탐색지역 재방문 횟수를 줄여 준다. branch and bound technique을 사용하여서 성능 향상을 가져 왔다. (추가 부가적인 설명들)

Instead of a blind exhaustive search or a branch and bound search, we propose selective search. We use the underlying image structure to generate object locations. 

> selective search는 blind exhaustive search나 branch and bound search대신에 image structure를 이용하여 물체의 위치 정보를 생성한다. 

In contrast to the discussed methods, this yields a completely class-independent set of locations. 

> 기존 방식과 비교 하면 이러한 방법을 사용함으로써 class-independent한 위치 정보들을 생성 할수 있다. 

Furthermore, because we do not use a fixed aspect ratio,our method is not limited to objects but should be able to find stuff like “grass” and “sand” as well (this also holds for [17]). 

> 더구나 제안 방안은 고정된 aspect ratio를 사용하지 않기 때문에 물체에 제약이 없다 또한 “grass” 나 “sand” 같은 탐지가 가능하다. 

Finally,we hope to generate fewer locations, which should make the problem easier as the variability of samples becomes lower. 

> ???

And more importantly, it frees up computational power which can be used for stronger machine learning techniques and more powerful appearance models.

> 계산 부하에 자유롭기 때문에 다른 부분(머신러닝 기술 &appearance model) 에서 계산부하가 큰 것을 써도 된다. 

### 2.2 Segmentation
Both Carreira and Sminchisescu [4] and Endres and Hoiem [9] propose to generate a set of class independent object hypotheses using segmentation. 

> [4][9]에서는 세그멘테이션을 사용하여서 클래스 독립적인 물체 예상 영역을 생성하는 방법을 제안 하였다. 

Both methods generate multiple foreground/background segmentations, learn to predict the likelihood that a foreground segment is a complete object, and use this to rank the segments.

> 두 방식 모두 여러개의 전경/배경 세그멘테이션들을 생성하여서 전경에는 물체가 있을 가능성이 높게 보고 세그멘테이션에 순위를 할당 하였다. 

Both algorithms show a promising ability to accurately delineate(윤곽을 그리다) objects within images, confirmed by [19] who achieve state-of-the-art results on pixel-wise image classification using [4].

> 두 방식 모두 이미지내 물체의 윤곽을 그리는데는 좋은 성과를 보였다. [19]에서는 [4]를 이용하여서 가장 최신의 결과를 보였다. 

As common in segmentation, both methods rely on a single strong algorithm for identifying good regions. 

> 두 방식 모두 강력한 하나의 알고리즘만을 사용하여 영역을 탐지 하고 있다. . 


They obtain a variety of locations by using many randomly initialised foreground and background seeds. 

> 두 방식은 무작위로 초기화한  전경/배경 seed정보를 이용하여 많은 위치 정보를 얻은다. 

In contrast, we explicitly deal with a variety of image conditions by using different grouping criteria and different representations.

> 이와 반대로 우리는 different grouping criteria 와를 이용하여서  different representations 다양한 이미지의 상황을 고려 하여 작업을 진행한다. 

This means a lower computational investment as we do not have to invest in the single best segmentation strategy, such as using the excellent yet expensive contour detector of [3]. 

> 이말은 ????

Furthermore,as we deal with different image conditions separately, we expect our locations to have a more consistent quality. 

> 더불어, 우리 장식은 서로 다른 이미지들의 상황을 분리적으로 다룸으로써 위치정보 또한 질이 좋을것으로 생각 한다. 

Finally, our selective search paradigm dictates that the most interesting question is not how our regions compare to [4, 9], but rather how they can complement each other.

> 최종적으로 우리 제안의 목적은 [4][9]와 대비하여 얼마나 좋은가가 아니라 어떻게 상호 보완할수 있는가 이다. 


Gu et al. [15] address the problem of carefully segmenting and recognizing objects based on their parts. 

? [15]는 their parts에 기반한 세그멘팅과 물체 탐지 기법의문제점에 대하여 기술 하였다. 

They first generate a set of part hypotheses using a grouping method based on Arbelaez etal. [3]. 

Each part hypothesis is described by both appearance and shape features. 

Then, an object is recognized and carefully delineated by using its parts, achieving good results for shape recognition.

In their work, the segmentation is hierarchical and yields segments at all scales. 

However, they use a single grouping strategy  whose power of discovering parts or objects is left unevaluated. 

In this work, we use multiple complementary strategies to deal with as many image conditions as possible. 
 
We include the locations generated using [3] in our evaluation.

### 2.3 Other Sampling Strategies

Alexe et al. [2] address the problem of the large sampling space of an exhaustive search by proposing to search for any object, independent of its class. 
In their method they train a classifier on the object windows of those objects which have a well-defined shape(as opposed to stuff like “grass” and “sand”). 
Then instead of a full exhaustive search they randomly sample boxes to which they apply their classifier. 
The boxes with the highest “objectness” measure serve as a set of object hypotheses. 
This set is then used to greatly reduce the number of windows evaluated by class-specific object detectors. 
We compare our method with their work.

> [2]에서는 이미지에서 objectnessr가 높은 영역을 간추리고 나서 식별자를 사용하하였다. 

Another strategy is to use visual words of the Bag-of-Words model to predict the object location. 
Vedaldi et al. [34] use jumping windows [5], in which the relation between individual visual words and the object location is learned to predict the object location in new images. 

Maji and Malik [23] combine multiple of these relations to predict the object location using a Hough-transform, after which they randomly sample windows close to the Hough maximum.In contrast to learning, we use the image structure to sample a set of class-independent object hypotheses.

> 또다른 방법은 "Bag-of-Words model"를 사용하는 것이다. 

To summarize, our novelty is as follows. 
Instead of an exhaustive search [8, 12, 16, 36] we use segmentation as selective search yielding a small set of class independent object locations. 

In contrast to the segmentation of [4, 9], instead of focusing on the best segmentation algorithm [3], we use a variety of strategies to deal with as many image conditions as possible, thereby severely reducing computational costs while potentially capturing more objects accurately. 

Instead of learning an objectness measure on randomly sampled boxes [2], we use a bottom-up grouping procedure to generate good object locations

> 우리 제안 방식의 새로운점들은 아래와 같다. 
> - exhaustive search대신에 segmentation 를 사용하여 분류와 독립적인 물체 위치 정보를 획득 하였다. 
> - 최고의 하나의 알고리즘에 초점을 두고 있는 [4][9]와는 이미지의 상황을 고려 하기 위하여 많은 전략들을 모두 고려 하였다.  
> - bottom-up grouping procedure 를 사용하여 물체 위치정보를 생성 하였다. 

## 3. Selective Search
In this section we detail our selective search algorithm for object recognition and present a variety of diversification strategies to deal with as many image conditions as possible. 

> 본장에서는 제안 방식에 대해 좀더 살펴 보고 이미지 상황을 고려 하기 위한 다양한 전략들을 살펴 보겠다. 

A selective search algorithm is subject to the following design considerations:
> 제안 알고리즘 설계시 고려 사항은 아래와 같다. 

Capture All Scales. (모든 크기를 커버 할것)
- Objects can occur at any scale within the image.Furthermore, some objects have less clear boundaries then other objects. 
- Therefore, in selective search all object scales have to be taken into account, as illustrated in Figure2. This is most naturally achieved by using an hierarchical algorithm.

> hierarchical algorithm을 이용하여 모든 크기의 물체 탐지 가능 

Diversification. (다양성)
- There is no single optimal strategy to group regions together. 
- As observed earlier in Figure 1, regions may form an object because of only colour, only texture, or because parts are enclosed. 
- Furthermore, lighting conditions such as shading and the colour of the light may influence how regions form an object. 
- Therefore instead of a single strategy which works well in most cases, we want to have a diverse set of strategies to deal with all cases.

> 하나의 전략으로도 잘 되긴 하지만, 다양한 전략을 사용 하도록 설계 

Fast to Compute. (빠른 속도)
- The goal of selective search is to yield a set of possible object locations for use in a practical object recognition framework. 
- The creation of this set should not become a computational bottleneck, hence our algorithm should be reasonably fast.

### 3.1 Selective Search by Hierarchical Grouping
We take a hierarchical grouping algorithm to form the basis of our
selective search. Bottom-up grouping is a popular approach to segmentation
[6, 13], hence we adapt it for selective search. Because
the process of grouping itself is hierarchical, we can naturally generate
locations at all scales by continuing the grouping process until
the whole image becomes a single region. This satisfies the condition
of capturing all scales.
As regions can yield richer information than pixels, we want to
use region-based features whenever possible. To get a set of small
starting regions which ideally do not span multiple objects, we use3.1 Selective Search by Hierarchical Grouping
We take a hierarchical grouping algorithm to form the basis of our
selective search. Bottom-up grouping is a popular approach to segmentation
[6, 13], hence we adapt it for selective search. Because
the process of grouping itself is hierarchical, we can naturally generate
locations at all scales by continuing the grouping process until
the whole image becomes a single region. This satisfies the condition
of capturing all scales.
As regions can yield richer information than pixels, we want to
use region-based features whenever possible. To get a set of small
starting regions which ideally do not span multiple objects, we use