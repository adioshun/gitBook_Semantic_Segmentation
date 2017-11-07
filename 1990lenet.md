|논문명 | |
| --- | --- |
| 저자\(소속\) | \(\) |
| 학회/년도 | IROS 2015, [논문]() |
| 키워드 | |
| 데이터셋(센서)/모델 | |
| 관련연구||
| 참고 | |
| 코드 | |





#LeNet

> Gradient-based learning applied to document recognition, LeCun, 1998


Yann LeCun이 1990년대에 발표한 구조로, 처음으로 CNN이라는 개념을 성공적으로 도입하였으며,

주로 우편 번호나 숫자 등을 인식하기 위해서 개발이 되었다.

이미 앞선 class를 통해, 어느 정도 그 구조에 대하여 살펴보았으며, CNN역사에 있어 古典이 되고 있다.


![](http://i.imgur.com/thmMaEF.png)

- C는 convolution, S는 sub-sampling, F는 Fully-connected layer를 의미하며,
- 대문자 알파벳 다음에 오는 소문자 x는 layer의 번호를 나타낸다
- 입력 : 32x32 영상
- 구조 : 3개의 convolution layer, 2개의 sub-sampling layer 및 1개의 fully-connected layer

## C1

- C1 은 convolutional layer이며 32x32 영상을 입력으로 받아,28x28 크기의 feature-map 영상을 만들어 낸다.

- 5x5 kernel을 사용하고, zero-padding을 지원하지 않기 때문에 boundary 정보가 사라지면서 28x28 크기의 feature-map 영상이 나오게 된다.

>(입력이미지 크기 + 양면(2) x pad-필터 크기)/Strid + 1 = (32 + 2 x 0 - 5)/1 + 1 = 28 ⇒ 32x32


C1 단계는 각 convolution kernel에서 (총 26 = 25 +1)의 자유 파라미터가 있고,

그런 커널이 6개 있기 때문에 총 156개의 자유 파라미터가 있다.




## S2

- S2는 sub-sampling을 수행하며, 2x2 크기의 receptive field로부터 average pooling을 수행하기 때문에, 
결과적으로28x28 크기의 feature-map 영상을 입력으로 받아, 14x14 크기의 출력 영상을 만들어 내며,각각의 feature map에 대해 1개의 대응하는 sub-sampling layer가 있다.

- Average pooling을 수행하기 때문에 weight 1 + bias 1 로 각각의 sub-sampling layer는 2개의 파라미터를 갖고, 자유 파라미터의 개수는 총 12개 있다.




## C3
- C3는 C1과 동일한 크기의 5x5 convolution을 수행하며, 14x14 입력 영상을 받아 10x10 크기의 출력 영상을 만들어 낸다.

- 6개의 입력 영상으로부터 16개의 convolution 영상을 만들어 내는데, 이 때 6개의 모든 입력 영상이 16개의 모든 출력 영상에 연결이 되는 것이 아니라, 선택적으로 입력 영상을 골라, 출력 영상에 반영이 될 수 있도록 한다.
 - 이렇게 하는 이유는 연산량의 크기를 줄이려는 이유도 있지만, 결정적인 이유는 연결의 symmetry를 깨줌으로써, 처음 convolution으로부터 얻은 6개의 low-level feature가 서로 다른 조합으로 섞이면서 global feature로 나타나기를 기대하기 때문이다.

- 이 단계의 자유 파라미터의 개수는 1516개이다. 이는 25(커널) x 60(S2의 feature map과 C3의 convolution에 대한 연결) + 16(bias)의 결과이다.


## S4
S4는 S2와 마찬가지로 sub-sampling 단계이며,

10x10 feature-map 영상을 받아 5x5 출력 영상을 만들며, 이 단계의 자유 파라미터의 개수는 32(2x16)개 이다.




## C5
C5는 단계는 16개의 5x5 영상을 받아, 5x5 kernel 크기의 convolution을 수행하기 때문에

출력은 1x1 크기의 feature-map이며,

이것들을 fully connected의 형태로 연결하여 총 120개의 feature map을 생성한다.

이전 단계에서 얻어진 16개의 feature-map이 convolution을 거치면서 다시 전체적으로 섞이는 결과를 내게 된다.


## F6

F6는 fully-connected이며 C5의 결과를 84개의 unit에 연결을 시킨다.

자유 파라미터의 개수는 (120+1)x84 = 10164가 된다.


입력 영상에 대해, 각각의 단계별 영상은 아래 그림과 같다.