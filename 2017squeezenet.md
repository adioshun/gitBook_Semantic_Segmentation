|논문명|Squeeze-and-Excitation Networks|
|-|-|
|저자(소속)|Jie Hu, Li Shen, Gang Sun|
|학회/년도|2017, [논문](https://arxiv.org/abs/1709.01507)|
|키워드|ILSVRC 2017 1위|
|참고||
|Code|[TF](https://github.com/taki0112/SENet-Tensorflow), [pyTorch](https://github.com/Queequeg92/SE-Net-CIFAR) |


# SENet

간략하게 요약을 하자면, resnet이나 densenet처럼 획기적인 네트워크를 제안한것은 아니고,
Squeeze and Excitation이란 block을 기존의 residual network나 inception network 등에 새롭게 추가를 하면, 성능이 더 올라간다 라는 네트워크입니다.
성능이 엄청나게 올라가진 않았지만, 이러한 block을 추가하면 기존 네트워크보다 성능이 다 올라간다. 라는것이 핵심인거같습니다.
논문상에서는 ResNeXt, Inception-v4등에 추가를 했고, Densenet에는 추가를 안했네요. (아마 densenet 자체가 너무 복잡해서 그런거같습니다.)