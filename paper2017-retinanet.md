|논문명|Focal Loss for Dense Object Detection|
|-|-|
|저자(소속)|Ross Girshick,|
|학회/년도| [논문](https://arxiv.org/abs/1708.02002)|
|키워드|Focal Loss, SSD계열 |
|참고|[ppt](https://www.slideshare.net/ssuser06e0c5/focal-loss-detection-classification/ssuser06e0c5/focal-loss-detection-classification)|
|Code||


Detector 중 One-stage Network(YOLO, SSD 등)의 문제점 예를 들어 근본적인 문제인 # of Hard positives(object) << # of Easy negatives(back ground) 또는 large object 와 small object 를 동시에 detect하는 경우 등과 같이 극단적인 Class 간 unbalance나 난이도에서 차이가 나는 문제가 동시에 존재함으로써 발생하는 문제를 해결하기 위하여 제시된 Focal loss를 class간 아주 극단적인 unbalance data에 대한 classification 문제(예를 들어 1:10이나 1:100)에 적용한 실험결과가 있어서 정리해봤습니다. 결과적으로 hyper parameter의 설정에 매우 민감하다는 실험결과와 잘만 활용할 경우, class간 unbalance를 해결하기 위한 data level의 sampling 방법이나 classifier level에서의 특별한 고려 없이 좋은 결과를 얻을 수 있다는 내용입니다.

