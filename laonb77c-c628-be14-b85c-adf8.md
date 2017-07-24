> 출처 : [라온피플 블로그](http://laonple.blog.me)

# Image Segmentation - "기본 이론

## 1. 개요 

### 1.1 집단화의 법칙(Law of Grouping)

영상 속에 있는 개체를 집단화할 수 있는 5개 법칙

1923년 베르트하이머가 체계화 시켰다.


- 유사성(Similarity): 모양이나 크기, 색상 등 유사한 시각 요소들끼리 그룹을 지어 하나의 패턴으로 보려는 경향이 있으며, 다른 요인이 동일하다면 유사성에 따라 형태는 집단화.

- 근접성(Proximity): 시공간적으로 서로 가까이 있는 것들을 함께 집단화해서 보는 경향.

- 공통성(Commonality): 같은 방향으로 움직이거나 같은 영역에 있는 것들을 하나의 단위로 인식하며, 배열이나 성질이 같은 것들끼리 집단화 되어 보이는 경향.

- 연속성(Continuity): 요소들이 부드럽게 연결될 수 있도록 직선 혹은 곡선으로 서로 묶여 지각되는 경향.

- 통폐합(Closure): 기존 지식을 바탕으로 완성되지 않은 형태를 완성시켜 지각하는 경향.

![](http://postfiles2.naver.net/MjAxNjExMjlfMTY3/MDAxNDgwMzkwNDc5NDE5.PlR1IgkZDSez9PkWvmv0rRYgs81Pck45iJFNLCabZmog.syrgZBGUov5FdNMdBUgAWXIc2iLERuQjMb4CH-8wGVsg.PNG.laonple/%EC%9D%B4%EB%AF%B8%EC%A7%80_2.png?type=w2)


### 1.2 Segmentation 방법 

영상을 구성하고 있는 주요 성분으로 분해를 하고, 관심인 객체의 위치와 외곽선을 추출

Segmentation의 접근 방법에 따라 크게 3가지 방식으로 분류가 가능

#### A. 픽셀 기반 방법 
이 방법은 흔히 thresholding에 기반한 방식으로 histogram을 이용해 픽셀들의 분포를 확인한 후 적절한 threshold를 설정하고, 픽셀 단위 연산을 통해 픽셀 별로 나누는 방식이며, 이진화에 많이 사용이 된다. 

thresholding으로는 전역(global) 혹은 지역(local)로 적용하는 영역에 따른 구분도 가능하고, 적응적(adaptive) 혹은 고정(fixed) 방식으로 경계값을 설정하는 방식에 따른 구별도 가능하다.

> 상세내용 참고 :[Image Segmentation - "기본 이론(Part 2)"](http://laonple.blog.me/220874313327)

#### B. Edge 기반 방법
Edge를 추출하는 필터 등을 사용하여 영상으로부터 경계를 추출하고, 흔히 non-maximum suppression과 같은 방식을 사용하여 의미 있는 edge와 없는 edge를 구별하는 방식을 사용한다.

> 상세내용 참고 :Image Segmentation - ["기본 이론(Part 3)"](http://laonple.blog.me/220875555860), ["기본 이론(Part 4)-Canny 알고리즘"](http://laonple.blog.me/220876492301), ["기본 이론(Part 5)"-SUSAN 알고리즘](http://laonple.blog.me/220885732170)



#### C. 영역 기반 방법 (자세히 다룸)
Thresholding이나 Edge에 기반한 방식으로는 의미 있는 영역으로 구별하는 것이 쉽지 않으며, 특히 잡음이 있는 환경에서 결과가 좋지 못하다. 

하지만 영역 기반의 방법은 기본적으로 영역의 동질성(homogeneity)에 기반하고 있기 때문에 다른 방법보다 의미 있는 영역으로 나누는데 적합하지만 동질성을 규정하는 rule을 어떻게 정할 것인가가 관건이 된다. 

흔히 seed라고 부르는 몇 개의 픽셀에서 시작하여 영역을 넓혀가는 region growing 방식이 여기에 해당이 된다. 이외에도 region merging, region splitting, split and merge, watershed 방식 등도 있다.

|Region을 결정하는 방식|
|-|
|- Region Growing<br>- Region Merging <br>- Region Splitting<br>- Split and Merge<br>- Watershed|

###### Edge기반  Vs. 영역기반 비교 
엣지 기반 segmentation 방법은 영상에서 차이(difference)가 나는 부분에 집중을 하였다면, 
영역(region) 기반의 segmentation 방법은 영상에서 **비슷한 속성**을 ﻿갖는 부분(similarity)에 집중하는 방식이며, 

엣지 기반 방식이 outside-in 방식이라고 한다면 
영역 기반의 방식은 inside-out 방식이라고 볼 수 있다.

> 비슷한 속성 : 밝기(intensity, gray-level), 컬러, 표면의 무늬(texture) 등.

##### 가. Region Growing 알고리즘

영역 기반의 방식에서 가장 많이 사용되는 방식이 region-growing 알고리즘이다. 

이 방식은 기준 픽셀을 정하고 기준 픽셀과 비슷한 속성을 갖는 픽셀로 영역을 확장하여 더 이상 같은 속성을 갖는 것들이 없으면 확장을 마치는 방식이다.

![](http://postfiles13.naver.net/MjAxNjEyMjBfMjUz/MDAxNDgyMTkzMjM0Nzg5._ucYSdyPr9bDftTP2Wffrn-JOLfAOQk13OY29nHDsMog.P-671qUZG46Mn0MgeO21bRTw0RKagzV9-axx5My9a3Ag.PNG.laonple/%EC%9D%B4%EB%AF%B8%EC%A7%80_22.png?type=w2)

아래 그림은 꽃잎 부분을 같은 영역이라고 찾아내는 예를 보여준다. 
- 먼저 임의의 픽셀을 seed로 정한 후 같은 속성(아래의 예에서는 컬러)을 갖는 부분으로 확장을 해나가면 오른쪽처럼 최종 영역을 구별해 낼 수 있게 된다. 
- 그림에 있는 다른 영역들도 비슷한 방식을 적용하면 구별이 가능해지게 된다.

###### seed(시작 픽셀)를 정하는 방식 
- 사전에 사용자가 seed 위치를 지정
- 모든 픽셀을 seed라고 가정
- 무작위로 seed 위치를 지정

###### seed 픽셀로부터 영역을 확장하는 방식

- 원래의 seed 픽셀과 비교: 
    - 영역 확장 시 원래의 seed 픽셀과 비교하여 일정 범위 이내가 되면 영역을 확장하는 방법. 
    - 잡음에 민감하고, seed를 어느 것으로 잡느냐에 따라 결과가 달라지는 경향이 있음.
  
      
- 확장된 위치의 픽셀과 비교: 
    - 원래의 seed 위치가 아니라 영역이 커지면 비교할 픽셀의 위치가 커지는 방향에 따라 바뀌는 방식. 
    - 장점은 조금씩 값이 변하는 위치에 있더라도 같은 영역으로 판단이 되나, 한쪽으로만 픽셀값의 변화가 생기게 되면 seed와 멀리 있는 픽셀은 값 차이가 많이 나더라도 (심각한drift 현상) 같은 영역으로 처리될 수 있음. 


- 영역의 통계적 특성과 비교: 
    - 새로운 픽셀이 추가될 때마다 새로운 픽셀까지 고려한 영역의 통계적 특성(예를 들면 평균)과 비교하여 새로운 픽셀을 영역에 추가할 것인지를 결정. 
    - 영역 내에 포함된 다른 픽셀들이 완충작용을 해주기 때문에 약간의 drift는 있을 수 있지만 안전. “centroid region growing” 이라고도 함.


|장점|단점|
|-|-|
|- 처리 속도가 빠르다.<br>- 개념적으로 단순하다.<br>- Seed 위치와 영역 확장을 위한 기준 설정을 선택할 수 있다.<br>- 동시에 여러 개의 기준을 설정할 수도 있다.|- 영상의 전체를 보는 것이 아니라 일부만 보는 “지역적 방식(local method)”이다<br>- 알고리즘이 잡음에 민감하다.<br>- Seed 픽셀과 비교하는 방식이 아니면 drift 현상이 발생할 수 있다.|


##### 나. Region Merging

비슷한 속성을 갖는 영역들을 결합시켜 동일한 꼬리표(label)를 달아주는 방식이다.

region merging은 어떤 경우는 매 픽셀 단위가 될 수 있으며, 일반적으로 심하게 나뉜 영역(over-segmented region)을 시작점으로 하며, 일반적으로 아래와 같은 과정을 거친다. 

1. 인접 영역을 정한다.

2. 비슷한 속성인지를 판단할 수 있는 통계적인 방법을 적용하여 비교한다.

3. 같은 객체라고 판단이 되면 합치고, 다시 통계를 갱신한다.

4. 더 이상 합칠 것이 없을 때까지 위 과정을 반복한다.

> Region growing은 region merging 방법 중 하나이며, 
> - region growing 방법은 1개 혹은 적은 수의 seed를 사용하는 방식으로 픽셀 단위로 판단을 하는 점만 차이가 있다. 
>- 반면에 merging은 영역을 기본 단위로 하며, 물론 가장 작은 영역은 픽셀이기 때문에 픽셀을 기본 영역으로 볼 수 있음, 그림 전체에 여러 개의 seed를 사용한다고 볼 수 있다.

##### 다. Region Splitting

영역 분리(region splitting) 방식은 merging과는 정반대 개념이라고 보면 된다. 

그림 전체와 같은 큰 영역을 속성이 일정 기준을 벗어나면 쪼개면서 세분화된 영역으로 나누는 방식을 사용한다.

보통은 아래 그림과 같이 4개의 동일한 크기를 갖는 영역으로 나누기 때문에 quad-tree splitting 방식을 많이 사용한다.

- 아래 그림은 큰 영역을 먼저 4개의 영역으로 나누고, 다시 각 영역을 검토하여 추가로 나눠야 할 것인지를 결정한다. 
- 그림에서는 2번 영역이 추가로 나눠야 하기 때문에, 21부터 24까지 다시 4개의 영역으로 나눈다. 다시 검토를 해보니 23영역을 4개의 영역으로 나눈 것이다.

 
![](http://postfiles2.naver.net/MjAxNjEyMjZfMzUg/MDAxNDgyNzUzNDk5ODUx.at_o74ghy4a8OY_k5eO_9nfnTdbENADF-3Mt2KGSmJAg.D578P3sfmpw4e8qgL8jHUm5TWoo4AK75MBASnXQ8ZvQg.PNG.laonple/%EC%9D%B4%EB%AF%B8%EC%A7%80_27.png?type=w2)



이렇게 region splitting 방식에서는 해당 영역에서 “분산(variance)”이나 “최대값과 최소값의 차”와 같은 통계 방식의 일정 기준을 설정하고, 그 값이 미리 정한 임계 범위를 초과하게 되면 영역을 분할한다.

##### 라. Split & Merge 

앞서 살펴본 것과 같이, splitting 방식만으로는 원하는 결과를 얻기가 어렵기 때문에 동일한 영역을 다시 합쳐주는 과정을 거쳐야 한다. 이것이 바로 split & merge 이다. 

Quad-tree splitting 방식을 사용하면 실제로는 모양이 위 그림과 같더라도 여러 개의 영역으로 분리될 수 밖에 없기 때문에 over-segmentation이 일어난다. 
- 이것을 같은 특성을 갖는 부분끼리 다시 묶어주면 {1, 3, 233}과 {4, 21, 22, 24, 231, 232, 234} 두 개의 영역으로 된다

![](http://postfiles14.naver.net/MjAxNjEyMjZfMTI4/MDAxNDgyNzUzNTI0OTI1.X9YK0IV1P2BVaXyqqoQHy-jUWK6n8JEnwEQk9v4oYegg.u1UzhVYBHUymopan-TVJgTeLFOAPNBKl2lIKfQLzZ5gg.PNG.laonple/%EC%9D%B4%EB%AF%B8%EC%A7%80_29.png?type=w2)

> region merging과 split & merge의 차이점
> - 차이점은 split & merge가 좀 더 속도가 빠르다는 점이다. 
> - 단순 merge가 막고 품는 것과 같다면, quad-tree splitting을 적용하면, 비슷한 영역은 통으로 처리가 되기 때문에 좀 더 속도가 빠르다. 

##### 마.  watershed segmentation

- region growing, merging 및 splitting : 비슷한 속성에 주목
- Watershed segmentation: 지형학에서 사용하는 개념도입 

> “Use of Watershed in Contour Detection” , Serge Beucher, Christian Lantuej, 1979 : 의료분야에서 많이 사용


![](http://postfiles12.naver.net/MjAxNzAxMDRfMjQx/MDAxNDgzNTI1Mzg3ODg2.n2mOUEPusW00jt_1Ii2MeEbZjQWlDkqQPth10ip4rRQg.9j8snqo7X2ryI9lbZhBnNM7vDnI2TBC5zu13i8eNBlcg.PNG.laonple/%EC%9D%B4%EB%AF%B8%EC%A7%80_1.png?type=w2)


Watershed는 산등성이나 능선처럼 비가 내리면 양쪽으로 흘러 내리는 경계에 해당이 되며, Catchment Basin은 물이 흘러 모이는 집수구역에 해당이 된다. Watershed는 기본적으로 영역을 구분해주는 역할을 하기 때문에 Watershed만 구하면 영상을 segmentation 할 수 있게 된다.

> 상세내용 참고 :Image Segmentation - ["기본 이론(Part 8)"](http://laonple.blog.me/220902777415)


### 1.3 Segmentation의 과정

Segmentation 과정은 bottom-up 방식과 top-down 방식이 있다. 

- Bottom-up 방식은 비슷한 특징을 갖는 것들끼리 집단화 하는 것을 말하며, 
- top-down 방식은 같은 객체에 해당하는 것들끼리 집단화 하는 것을 말한다. 

![](http://postfiles2.naver.net/MjAxNjExMjlfNTAg/MDAxNDgwMzkwNDc4NjQ3.4JfEae59AIvHBdNVWN-XfHnVS7RCAtW1nkAiTj8aN6Qg.yybBAXG41AWLIZ2GjIH4slsOFcm4xii0xKYVFKM3PPQg.PNG.laonple/%EC%9D%B4%EB%AF%B8%EC%A7%80_4.png?type=w2)

