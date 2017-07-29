# Pooling Layer

부제 : Semantic Segmentation의 골치 꺼리 Pooling

## 1. 정의 
목적 : 위치 변화에 대한 강건성 확보 
    - 합성곱층에서 추출한 자질의 위치 감도를 약간 저하시켜 대상이 되는 자질값의 이미지 내에서의 위치가 조금씩 변화하는 경우에도 풀링층의 출력에는 변화를 주지 않는다. 

위치 : 보통 합성층 바로 뒤 

종류 
- 최대 풀링 : $$H^2$$개의 픽셀값 중 최댓값을 고르기 
- 평균 풀링 : 평균값을 계산한후 픽셀값으로 나눈다. 

$$
u_{ijk}=\left( \frac{1}{H^2} \sum_{(p,q)\in P_{ij}z^P_{pqk}} \right)^\frac{1}{P} ...\begin{cases}P=1  :평균풀핑\\ P=\infty : 최대 풀링 \end{cases}
$$

계산 
- 입력 이미지의 각 채널마다 병렬로 실행 
    - 따라서, 입력 채널수와 출력 채널수는 같다. 
- Stride 설정 가능
- Pad 설정 가능 

## 2. 다양한 Pooling Layer

### 2.1 Spatial Pyramid Pooling
> SPPNet에 적용된 Pooling Layer

![](http://i.imgur.com/IPbiLQ3.png)

- Object Detection 네트워크의 후보영역(다양한크기)에 대한 처리 
    - 다양한 입력 크기 처리
    - 위치 정보 유지 


#### A. 다양한 입력 크기 처리 

- 기존 : VGG16등의 활용을 위해 지정된 크기로 Croping 수행 (이미지 외곡 발생)

- 제안 Bag-of-words(BoG)를 활용하여 다양한 크기의 (후보영역)입력으로부터 일정한 크기(VGG16입력)의 출력 feature를 추출  
    - BoG : 많은 입력 패턴 중 강한 특징 성분(Top x)이 무었인지 확인

- 단점 : 위치 정보 사라짐 

### B. 위치 정보 유지 

- 기존 : BoG 사용시 위치 정보는 사라지고 많은 입력 패턴 중 가장 강한 특징 성분이 무었인지만 확인 가능 [[1]](https://adioshun.gitbooks.io/semantic-segmentation/content/selective-search2012.html)

- 제안 : 위치별로 BoG를 실시 하여 위치 정보 유지 
    - 나누는 위치의 갯수는 이후 Fully Connected Layer에 입력 요구사항으로 결정 (맞나??)

![](http://i.imgur.com/xZde5hv.png)

> 출처 : [R-CNNs Tutorial](https://blog.lunit.io/2017/06/01/r-cnns-tutorial/)

### 2.2 Region of Interest(RoI) Pooling
>Fast R-CNN에 적용된 Pooling Layer

![](http://i.imgur.com/idoUX2g.png)

- SPP layer[2.1]의 single level pyramid만을 사용

- RoI Pooling layer에서 다양한 후보 영역들에 대하여 FC layer로 들어갈 수 있도록 크기를 조정하는 역할 수행

#### A. 동작 과정 

목표 : 입력 feature map의 사이즈가 a $$\times$$ a 이고, 출력 bin이  n $$\times$$ n개로 함 

방법 : bin의 window size를 올림하고( $$win = \lceil a/n \rceil $$), 각 bin을 pooling하기 위한 stride를 내림 ($$ str = \lfloor a/n \rfloor$$ ) 

장점 : bin의 window size가 정수배가 되지 않는 misalignment문제 해결 (eg.  13×13 feature map을 6×6 bins으로 나눌 경우 window size가 2.167)

단점 : 올림/내림 과정에서 좌표값의 미묘한 차이 발생 
    - Fast R-CNN 같은 Object Detection에는 큰 문제 아님 
    - Mask R-CNN 같은 Fixel단위 Detection에는 큰 문제, [2.3]으로 해결
    
### 2.3 RoIAlign layer

> Mask R-CNN에 적용된 Pooling Layer

- Fast R-CNN에서 사용하는 RoI Pooling layer의 단점 해결 
