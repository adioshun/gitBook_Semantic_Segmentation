|논문명 | |
| --- | --- |
| 저자\(소속\) | \(\) |
| 학회/년도 | IROS 2015, [논문]() |
| 키워드 | |
| 데이터셋(센서)/모델 | |
| 관련연구||
| 참고 | |
| 코드 | |





# GoogLeNet


ILSVRC 2014년 대회에서 우승을 한 구조로 구글의 Szegedy 등에 의해서 개발이 되었다.



## 1. 개요 

이전에 비하여 2014년 이후에는 Deep해졌다. 
- 2013 이전 : 10 Layer 미만
- 2014 이후 : GoogLeNet(22 layer), VGGNet(19 layer)

Layer는 깊어 졌지만, “Inception Module” 개념을 도입하여 파라미터 수를 대폭 줄일 수 있게 되었다.

![](http://i.imgur.com/TL8cPq2.png)



## 2. 특징 

### 2.1 Network In Network

>  “Min Lin”이 2013년에 발표한 “Network In Network” 논문 

#### A. Linear Conv. VS. MLP Conv. 

![](http://i.imgur.com/4n8XIcT.png)

기존 CNN 문제  
- Feature 추출시 filter의 특징이 linear하기 때문에 non-linear한 성질을 갖는 feature를 추출하기엔 어려움(??)
- 이를 극복하기 위해 feature-map의 개수를 늘려야 함 
- 연산량이 늘어남 

제안 해결책 : filter 대신에 MLP(Multi-Layer Perceptron)를 사용하여 feature를 추출

MLP 장점 
-  convolution kernel 보다는 non-linear 한 성질을 잘 활용할 수 있기 때문에 feature를 추출할 수 있는 능력이 우수
- 1x1 convolution을 사용하여 feature-map을 줄일 수 있음

> 1x1 convolution은 중요 개념 이므로 [2.2]에서 다시 설명 


#### B. CNN VS. NIN

![](http://i.imgur.com/1oWxvdX.png)
[ MLPconv layer를 3개 사용한 NIN ]

Fully-connected NN 대신에 최종단에 “Global average pooling”을 사용
- 효과적으로 feature-vector를 추출하였기에 추출된 vector 들에 대한 pooling 만으로도 충분
- Average pooling 만으로 classifier 역할을 할 수 있기 때문에 overfitting의 문제를 회피
    - classifier 역할을 하는 FCN은 전체 free parameter 중 90%를 가지고 있음 
    - 많은 파라미터는 오버피팅에 빠질 가능성유발, FCN이 없으므로 오버피팅 문제 해결 

### 2.2 1x1 convolution

- 목표 : 차원 축소 

- 방법 : Hebbian principle(Neurons that fire together, wire together)

- 효과 : 여러 개의 feature-map으로부터 비슷한 성질을 갖는 것들을 묶어낼 수 있고(압축??) 


![](http://i.imgur.com/ftNngAF.png)
위 그림에서 “C2 > C3”의 관계가 만들어지면, 차원을 줄이는 것과 같은 효과를 얻을 수 있음

###### 보충 설명 :1-Layer FCN(fully-connected neural network)라고 불리기도 함 
- fully-connected와 동일한 방식

만약에 입력 feature-map c2의 갯수가 4이고, 출력 feature-map c3의 갯수가 2인 경우를 가정해보면, 1x1 convolution은 아래 그림과 같이 표현할 수 있다.
![](http://i.imgur.com/4Gf5QDo.png)

- 결과적으로 보면 4개의 feature-map으로부터 입력을 받아, 학습을 통해 얻어진 learned parameter를 통해 4개의 feature-map이 2개의 feature-map으로 결정이 된다.

- 즉, 차원이 줄어들게 되며, 이를 통해 연산량을 절감하게 된다.

- 또한, neuron에는 활성함수로 RELU를 사용하게 되면, 추가로 non-linearity를 얻을 수 있는 이점도 있다.



## 3. 구조 (구글의 인셉션-Inception)

Local receptive field에서 더 다양한 feature를 추출하기 위해 여러 개의 convolution(파란색)을 병렬적으로 활용

||기본구조|확장 구조|
|-|-|-|
||![](http://i.imgur.com/MqoQtOS.png)|![](http://i.imgur.com/lEr9DHG.png)|
|구조|1x1 convolution, 3x3 및 5x5 convolution, 3x3 max pooling|1x1 convolution을 cascade 구조로 두고, <br>1x1 convolution을 통해 feature-map의 개수(차원)를 줄임|
|장점|다양한 scale의 feature를 추출하기에 적합한 구조|feature 추출을 위한 여러 scale을 확보하면서도, 연산량의 균형을 맞출 수 있음|
|단점| 3x3과 5x5 convolution은 연산량이 큼||

- convolution kernel(파란색): 다양한 scale의 feature를 효과적으로 추출
- 1x1 convolution layer(노란색): 연산량을 크게 경감시킬 수 있게 되어, 결과적으로 망의 넓이와 깊이를 증가시킬 수 있는 기반이 마련 되었다.


### 3.1 9개의 인셉션 모듈로 이루어진 GooLeNet

![](http://i.imgur.com/ZAIklUC.png)

- 파란색 : convolutional layer
- 빨간색 : max-pooling
- 노란색 : Softmax layer
- 녹색 : 기타 function
- 동그라미 : 인셉션 모듈
- 동그라미 위 숫자: 각 단계에서 얻어지는 feature-map의 수

![](http://i.imgur.com/eej3wgL.png)

- Patch size/stride: 커널의 크기와 stride 간격을 말한다. 최초의 convolution에 있는 7x7/2의 의미는 receptive field의 크기가 7x7인 filter를 2픽셀 간격으로 적용한다는 뜻이다.

-  Output size: 얻어지는 feature-map의 크기 및 개수를 나타낸다. 112x112x64의 의미는 224x224 크기의 이미지에 2픽셀 간격으로 7x7 filter를 적용하여 총 64개의 feature-map이 얻어졌다는 뜻이다.

-  Depth: 연속적인 convolution layer의 개수를 의미한다. 첫번째 convolution layer는 depth가 1이고, 두번째와 인셉션이 적용되어 있는 부분은 모두 2로 되어 있는 이유는 2개의 convolution을 연속적으로 적용하기 때문이다.

-  1x1: 1x1 convolution을 의미하며, 그 행에 있는 숫자는 1x1 convolution을 수행한 뒤 얻어지는 feature-map의 개수를 말한다. 첫번째 인셉션 3(a)의 #1x1 위치에 있는 숫자가 64인데 이것은 이전 layer의 192개 feature-map을 입력으로 받아 64개의 feature-map이 얻어졌다는 뜻이다. 즉, 192차원이 64차원으로 줄어들게 된다.

-  3x3 reduce: 이것은 3x3 convolution 앞쪽에 있는 1x1 convolution 을 의미하여 마찬가지로 인셉션 3(a)의 수를 보면 96이 있는데, 이것은 3x3 convolution을 수행하기 전에 192차원을 96차원으로 줄인 것을 의미한다.

-  3x3: 1x1 convolution에 의해 차원이 줄어든 feature map에 3x3 convolution을 적용한다. 인셉션 3(a)의 숫자 128은 최종적으로 1x1 convolution과 3x3 convolution을 연속으로 적용하여 128개의 feature-map이 얻어졌다는 뜻이다.

-  5x5 reduce: 해석 방법은 “#3x3 reduce”와 동일하다.

-  5x5: 해석 방법은 “#3x3”과 동일하다. #5x5는 좀 더 넓은 영역에 걸쳐 있는 feature를 추출하기 위한 용도로 인셉션 모듈에 적용이 되었다.

-  Pool/proj: 이 부분은 max-pooling과 max-pooling 뒤에 오는 1x1 convolution을 적용한 것을 의미한다. 인셉션 3(a) 열의 숫자 32 는 max-pooling과 1x1 convolution을 거쳐 총 32개의 feature-map이 얻어졌다는 뜻이다.

-  Params: 해당 layer에 있는 free parameter의 개수를 나타내며, 입출력 feature-map의 數에 비례한다. 인셉션 3(a) 열에 있는 숫자 159K는 총 256개의 feature-map을 만들기 위해 159K의 free-parameter가 적용되었다는 뜻이다.

-  Ops: 연산의 수를 나타낸다. 연산의 수는 feature-map의 수와 입출력 feature-map의 크기에 비례한다. 인셉션 3(a)의 단계에서는 총 128M의 연산을 수행한다.


위 설명에 따라 표에 있는 각각의 숫자들의 의미를 해석해 보면, GoogLeNet의 구조를 좀 더 친숙하게 이해할 수 있다.


인셉션 3(a)에는 256이라는 빨간색 숫자가 적혀 있는데, 이것은 인셉션 3(a)를 통해 총 256개의 feature-map이 만들어졌다는 뜻이며, 이것은 1x1 convolution을 통해 64개, 1x1과 3x3 연속 convolution을 통해 128개, 1x1과 5x5 연속 convolution을 통해 32개, max-pooling과 1x1 convolution을 통해 32개를 적용하여 도합 256개의 feature-map을 얻을 수 있게 되었다는 뜻이다.


3x3보다는 5x5 convolution을 통해 얻는 feature-map의 개수가 작은 이유는 5x5 convolution이 훨씬 연산량을 많이 필요로 하기 때문이며, 입력 이미지의 크기가 이미 28x28로 줄어든 상황에서는 3x3으로 얻을 수 있는 feature가 5x5로 얻을 수 있는 feature보다 많기 때문일 것이다.


만약에 3x3이나 5x5 convolution 앞에 1x1 convolution이 없다면 어떻게 되었을까? 3x3 convolution의 경우는 1x1 convolution을 통해 192개의 feature-map이 96개로 줄었기 때문에 50% 정도 연산량이 줄었으며, 5x5의 경우는 192개가 16개로 줄었기 때문에, 약 91.7% 정도 연산량이 줄어들게 되었다. 특히 5x5에서 연산량이 크게 줄었기 때문에, 1x1 convolution을 통한 차원 절감의 효과를 크게 누릴 수 있다.

### 3.2  Auxiliary classifier
> Training Deeper Convolutional Networks with Deep SuperVision(Liwei Wang, Chen-Yu Lee 등)


![](http://i.imgur.com/US6mcfS.png)

문제점 : 깊은 망구조로 인하여 vanishing gradient 문제가 발생 할수 있다. 
- 학습 속도가 아주 느려지거나 overfitting 문제 야기 

해결책 :  학습을 할 때는 Auxiliary classifier 이용, GoogLeNet에서는 위 화살표 두군대에 적용 


> 자세한 내용은 [라온피플 블로그](http://laonple.blog.me/220692793375) 참고

### 3.3 Factorizing Convolutions

큰 필터 크기를 갖는 convolution 커널을 인수 분해 하면, 작은 커널 여러 개로 구성된 deep network를 만들 수 있으며, 이렇게 되면 parameter의 수가 더 줄어들면서 망은 깊어지는 효과를 얻을 수 있다.

|Before|5x5 Convolution $$\rightarrow$$2개의 3x3 Convolution|After|
|-|-|-|
|![](http://i.imgur.com/3wjrfwY.png)|![](http://i.imgur.com/vTe8aGS.png)|![](http://i.imgur.com/KwDQunl.png)|
|Parameter = 25||Parameter =9+9=18|
eg. 7x7 convolution의 경우도 위 경우와 마찬가지로 3 단의 3x3 convolution으로 대응이 가능

> 자세한 내용은 [라온피플 블로그](http://laonple.blog.me/220692793375) 참고

--- 
## 4. 효과적으로 Layer 크기 줄이는 법 


![](http://i.imgur.com/5BE9sNx.png)
35x35x320 $$\rightarrow$$ 17x17x640 으로 줄이는 2가지 방법 

### 3.1  Pooling -> Inception
계산 부하 적음, Pooling 단계를 거치면서, 원 feature-map에 있는 숨어 있는 정 (representational concept)가 사라지게 될 가능성 있음

### 3.2 Inception -> Pooling
 Inception 적용하였기 때문에 계산 부하 4배, Pooling전에 Inception을 적용 하였기 떄문에 숨은 특징을 더 잘 찾아낼 가능성은 높아진다.

### 3.3 Szegedy(GoogLeNet 설계자중 한 명) 제안 방법 2가지 

> Rethinking the inception architecture for computer vision

![](http://i.imgur.com/XzGmkzA.png)

#### A. (왼쪽)

- 5x5, 3x3 convolution을 통해 local feature를 추출하면서 stride 2를 통해 크기가 줄고, 

- 또한 pooling layer를 통해서도 크기를 줄이고 그 결과를 결합하는 방식

#### B. (오른쪽) 

- 좀 더 단순한 방법, 효율성과 연산량의 절감을 동시에 달성

- stride 2를 갖는 convolution을 통해 320개의 feature-map을 추출하고 

- pooling layer를 통해 다시 320개의 feature-map을 추출



