|논문명|Object Detection Networks on Convolutional Feature Maps|
|-|-|
|저자(소속)|Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun (MS)|
|학회/년도|2016, [논문](https://arxiv.org/abs/1504.06066)|
|키워드| |
|참고||


# Faster R-CNN with ResNet

## 1. 개요

Faster R-CNN논문에서는 Pre-train 모델로  ZFNet과 VGG-16을 이용하였다. 

RestNet을 이용하여 구현 할수는 없을까? 
- object detection = feature extraction + object classifier 로 나누어서 문제 해결 

![](http://i.imgur.com/zUoZPsZ.png)
> NoC(Network on Conv feature map): CNN을 통해서 얻어진 feature를 기반으로 별도의 망을 통해서 처리한다는 의미로 명명 




### 1.1 Feature extraction

- ConvNet으로 구현하기 때문에 큰 문제 없이 ResNet으로 충분히 구현이 가능

### 1.2 Object classifier

전통적 Classifier 구현 방법 : MLP(Multi-Layer Perceptron)를 fc(fully connected layer) 2~3개로 구현을 하는 방법

제안 구현 방법 : 하나의 망으로 간주를 하고 어떨 때 최적의 성능이 나오는지를 여러 실험으로 확인

![](http://i.imgur.com/uZ78gcM.png)

실험 결과 

- fully connected layer 앞에 convolutional layer를 추가하면 결과가 좋아지며, 

- multi-scale을 대응하기 위해 인접한 scale에서의 feature map 결과를 선택하는 "Maxout" 방식을 지원

## 2. 구조 

![](http://i.imgur.com/Tuyt8Rr.png)

- Feature extractor 부분은 conv1, conv2_x, conv3_x, conv4_x를 이용하여 구현할 수 있다. 
 - 이렇게 되면, VGG16 때와 마찬가지로 feature map에서의 1 픽셀은 stride 16에 해당이 된다. 
 - 또한 이 layer들은 RPN(Region Proposal Network)과 Fast R-CNN Network 공유하여 사용하게 된다.

RPN 이후 과정은 Fast R-CNN 구조(아래 그림에서 회색 영역에 해당)를 구현을 하여야 한다.

이미 Con1 ~ Conv4는 feature를 추출하기 위한 용도로 사용을 했으니, 당연히 Conv5 부분과 Classifier 부분을 잘 활용하거나 변화를 시켜야 한다는 것은 감을 잡을 수 있다.

ConvNet을 거쳐 얻어진 feature map을 conv5_1에 보내기 전에 RoI pooling을 수행하며, Conv5_x 부분은 위 그림에서 FCs에 해당하는 부분을 수행을 하며, 최종단의 Classifier는 위 그림의 최종단으로 대체를 해주면, ResNet을 약간만 변형을 하면, object detection에 활용이 가능하다.
