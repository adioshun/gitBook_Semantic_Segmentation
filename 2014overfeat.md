|논문명/저자/소속|OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks|
|-|-|
|저자(소속)|Pierre Sermanet, Yann LeCun(NYU)|
|학회/년도|ICLR2014, [논문](https://arxiv.org/abs/1312.6229), [발표_동영](https://www.youtube.com/watch?v=3U-bZgKFS7g)|
|키워드||
|참고|[코드](https://github.com/sermanet/OverFeat), [요약자료#1(한글)](http://laonple.blog.me/220752877630), [요약자료#2(한글)](http://www.whydsp.org/294)|



# OverFeat

--- 
> 출처 : 라온피플 블로그


Dense evaluation :  CNN의 classifier로 흔히 사용되는 Fully-connected layer를 1x1 convolution 개념으로 사용하게 되면, 고정된 이미지뿐만 아니라 다양한 크기의 이미지를 sliding window 방식으로 처리할 수 있으며, feature extraction 블락의 맨 마지막 단에 오는 max-pooling layer의 전후 처리 방식을 조금만 바꾸면, 기존 CNN 방식보다 보다 조밀하게(dense) feature extraction, localization 및 detection을 수행할 수 있게 된다. 물론 multi-crop 방식보다 연산량 관점에서 매우 효율적이다.



## 1. 개요 

OverFeat과 다른 알고리즘 비교
- classification, localization 및 detection에 대한 통합 프레임워크
- 1-pass로 연산이 가능한 구조를 취하고 있기 때문에, R-CNN 보다 연산량 관점에서 효과적
    - cf. SPPNet 역시 1-pass 구조(Spatial Pyramid Pooling) > OverFeat(dense evaluation)
    
    