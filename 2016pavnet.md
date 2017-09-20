|논문명|PVANet: Lightweight Deep Neural Networks for Real-time Object Detection|
|-|-|
|저자(소속)|홍상훈 (인텔)|
|학회/년도|2016, [논문](https://arxiv.org/abs/1611.08588)|
|키워드||
|참고|[PR12](https://www.youtube.com/watch?v=TYDGTnxUGHQ&feature=share), [발표자료](https://www.slideshare.net/JinwonLee9/pr12-pva-net)|

PR12 논문읽기 모임에서 33번째로 발표한 PVANet: Lightweight Deep Neural Networks for Real-time Object Detection 입니다.
벌써 PR12 논문모임의 목표인 논문 100편 읽기의 1/3이 지났습니다.
제가 object detection 쪽 논문을 세번째 review하고 있는데요 이번에는 우리나라 분들이 쓰신 논문을 review해보았습니다 1저자이신 Sanghoon Hong님께서 만드신 slide를 대부분 사용하였구요... 
이 논문은 Faster R-CNN을 기반으로 하여, modified concatenated ReLU, inception structure, hyper feature concatenation을 사용하여 정확하면서도 빠른 object detection을 가능하게 하였습니다. 
concatenated ReLU는 CNN의 아래쪽 layer가 굉장히 기본적인 feature들을 학습하는데 이 feature들이 서로 대칭적인 경우가 많아서(왼쪽 -> 오른쪽, 오른쪽 -> 왼쪽 등등) 이를 이용하여 연산량은 줄이면서 feature map 수는 비슷하게 가져가는 방법입니다. 이 논문에서는 이 CReLU에 scale/bias 를 더하여 더 좋은 성능을 보여주었구요, 
inception 구조도 detection용 network에서 잘 사용하지 않았는데, inception 구조가 object detection에서도 훌륭한 성능을 낼 수 있다는 것을 보여주었습니다.
또한 hyper feature concatenation을 통해서 size가 다른 feature map들을 가져오고 이를 다시 1x1 convolution으로 잘 섞어서(?) RPN이나 detection network에서 사용하였습니다. 
이러한 방법들이 모여서 VOC2012 datatset에서 80%이상의 mAP를 낸 detection 알고리즘 중에 가장 적은 연산량을 갖는 알고리즘이 되었습니다. SSD나 R-FCN과 같은 network에 접목이 가능한 방식이기 때문에 응용분야도 많은 논문이라고 생각됩니다. 자세한 내용은 아래 영상을 통해 보실 수 있습니다^^