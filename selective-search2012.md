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



## 2. 1 Introduction

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

Specifically, we use a data-driven grouping based strategy where we increase diversity by using a variety of complementary grouping criteria and a variety of complementary colour spaces with different invariance properties. 

The set of locationsis obtained by combining the locations of these complementarypartitionings. 
Our goal is to generate a class-independent,data-driven, selective search strategy that generates a small set ofhigh-quality object locations.
