|논문명|DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs|
|-|-|
|저자(소속)|Liang-Chieh Chen (구글)|
|학회/년도|2016 [논문](https://arxiv.org/pdf/1606.00915.pdf)|
|키워드| |
|참고|[라온피플](http://laonple.blog.me/221000648527)|


# DeepLab v2

2015: DeepLab v1
2015: dilated convolution, DeepLab v1의 단점 개선
2016: DeepLab v1, multiple-scale에 대한 처리 방법 개선, dilated convolution보다 성능 좋음 



## 1. 개요 

semantic segmentation = DCNN(deep convolutional neural networks) + atrous convolution + fully connected CRF

### 1.1 Classification 기반 망을 semantic segmentation에 적용할 때의 문제점

- Classification 망 : 대상의 존재 여부에 집중, conv+pooling을 통해 강한 Feature들 추출 목적, detail보다는 global한 것에 집중

- semantic segmentation 망 : 픽셀 단위의 조밀한 예측이 필요, 분류망은 pooling등을 통해 feature-map의 크기가 줄어들기 때문에 detail 정보를 얻는데 어려움

### 1.2 detail 정보를 얻는데 어려운 문제 해결책

- FCN : skip layer를 사용하여 1/8, 1/16, 및 1/32 결과를 결합하여 detail이 줄어드는 문제를 보강

- dilated convolution/DeepLab : 망의 뒷 단에 있는 2개의 pooling layer를 제거하고, dilated convolution 혹은 atrous convolution을 사용하여 receptive field를 확장시키는 효과를 얻었으며, 1/8 크기까지만 줄이는 방법을 사용하여 detail이 사라지는 것을 커버

### 1.3 1/8까지만 사용 시 문제점

- Receptive field가 충분히 크지 않아 다양한 scale에 대응이 어렵다.

- 1/8정보를 bilinear interpolation을 통해서 원 영상의 크기로 키우면, 1/32 크기를 확장한 것보다 detail이 살아 있기는 하지만, 여전히 정교함이 떨어진다.


