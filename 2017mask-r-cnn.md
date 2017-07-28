|논문명|Mask R-CNN|
|-|-|
|저자(소속)|Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick (Facebook)|
|학회/년도|2017 [논문](https://arxiv.org/abs/1703.06870)|
|키워드| |
|참고|[코드](https://github.com/CharlesShang/FastMaskRCNN)|


# Mask R-CNN

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



