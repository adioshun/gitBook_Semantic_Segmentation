|논문명|Mask R-CNN|
|-|-|
|저자(소속)|Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick (Facebook)|
|학회/년도|2017 [논문](https://arxiv.org/abs/1703.06870)|
|키워드| |
|참고| [sogangori(K)](http://blog.naver.com/sogangori/221012300995), [Yuthon(E)](http://www.yuthon.com/2017/04/27/Notes-From-Faster-R-CNN-to-Mask-R-CNN/) |
|구현 코드|[PyTorch](https://github.com/felixgwu/mask_rcnn_pytorch), [TensorFlow](https://github.com/CharlesShang/FastMaskRCNN)|


# Mask R-CNN
---
[Faster R-CNN to Mask R-CNN](http://www.yuthon.com/2017/04/27/Notes-From-Faster-R-CNN-to-Mask-R-CNN/)


---
[딥러닝강사](http://blog.naver.com/sogangori/221012300995)


---
> [텐서플로우 블로그](https://tensorflow.blog/2017/06/05/from-r-cnn-to-mask-r-cnn/)

![](http://i.imgur.com/OBXTpkJ.png)

페이스북 AI 팀이 분할된 이미지를 마스킹하는 Mask R-CNN을 내놓았습니다. 

바이너리 마스크binary mask : Faster R-CNN에 각 픽셀이 오브젝트에 해당하는 것인지 아닌지를 마스킹하는 네트워크(CNN)

페이스북 팀은 정확한 픽셀 위치를 추출하기 위해 CNN을 통과하면서 RoIPool 영역의 위치에 생기는 소숫점 오차를 2D 선형보간법bilinear interpolation을 통해 감소시켰다고 합니다. 이를 RoIAlign이라고 합니다. 



---
> 출처 : [Donghyun 블로그](http://blog.naver.com/kangdonghyun/221006015797)

![](http://i.imgur.com/Lec4AlE.png)

결론은, Faster R-CNN + mask branch 이게 끝이다!

처음엔 어떻게 이게 가능하지 싶었는데, 읽어보니 엄청 특별한건 없었다.

1. Segmentation 응용, mask branch

- mask branch는 각 RoI별로 K개의 class에 대해서 m * m 의 binary output을 뱉어낸다.
- 얘의 Loss까지 포함하여 학습해서 더 좋은 결과를 얻는다.
- 이전가지의 segmentation 연구들은 한 픽셀마다 K개의 class가 compete 하도록 구성되었는데, 이 연구에서는 그러지 않았음 (이게 segmentation result의 key가 되었다!)

- FCN 사용한 mask branch
-- Learning은 정답 class에 대해서만 한다. (False learning은 하지 않는다)
--- Target class k에 대한 loss만 전체 loss function에 더해준다
-- RoIPool 대신 RoIAlign 도입하여 사용

---
> https://blog.lunit.io/2017/06/01/r-cnns-tutorial/

## RoIAlign layer

misalignment 문제 
- Fast R-CNN이나 SPPNet에서 RoI Pooling 또는 SPP를 수행할 때  feature map의 사이즈가 a $$\times$$ a 이고, 이를 n $$\times$$ n개의 bin으로 나눈다고 했을 때, 각 bin의 window size가 정수배가 되지 않을 경우가 있습니다. (예를 들어, 13×13 feature map을 6×6 bins으로 나눌 경우 window size가 2.167이 됩니다) 

- SPPNet의 저자는 이러한 




