|논문명|Selective Search for Object Recognition|
|-|-|
|저자(소속)|J.R.R. Uijlings|
|학회/년도|IJCV 2012, [논문](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)|
|키워드||
|참고|[발표자료#1](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/ssearch_schuyler.pdf),[발표자료#2](http://www.cs.cornell.edu/courses/cs7670/2014sp/slides/VisionSeminar14.pdf)|

# Selective Search for Object Recognition

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
- Therefore, in selective search all objectscales have to be taken into account, as illustrated in Figure2. 
This is most naturally achieved by using an hierarchicalalgorithm.

Diversification. (다양성)
- There is no single optimal strategy to group regionstogether. 
As observed earlier in Figure 1, regions mayform an object because of only colour, only texture, or becauseparts are enclosed. 
Furthermore, lighting conditions such asshading and the colour of the light may influence how regionsform an object. 
Therefore instead of a single strategy whichworks well in most cases, we want to have a diverse set ofstrategies to deal with all cases.

- Fast to Compute. (빠른 속도)
The goal of selective search is to yield a set ofpossible object locations for use in a practical object recognitionframework. 
The creation of this set should not become acomputational bottleneck, hence our algorithm should be reasonablyfast.