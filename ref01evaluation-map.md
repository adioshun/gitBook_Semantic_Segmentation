# mAP
- We use a metric called “mean average precision” (mAP)

- Compute average precision (AP) separately for each class, then average over classes

- A detection is a true positive if it has IoU with a ground-truth box greater than some threshold (usually 0.5) (mAP@0.5)

- Combine all detections from all test images to draw a precision / recall curve for each class; AP is area under the curve

- TL;DR mAP is a number from 0 to 100; high is good




  Mean Average Precision, 
- To calculate it for Object Detection, you calculate the average precision for each class in your data based on your model predictions. Average precision is related to the area under the precision-recall curve for a class. Then Taking the mean of these average individual-class-precision gives you the Mean Average Precision.

[[The PASCAL Visual Object Classes (VOC) Challenge]](http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf)


# IoU 

## 1. 정의 

|![](http://i.imgur.com/HXXb6WX.png)|![](http://i.imgur.com/LjtA6h8.png)|
|-|-|

> 출처 : [Intersection over Union (IoU) for object detection](http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)


$$
IoU = \frac{TP}{FP+TP+FN}
$$


## 2. IoU 를 학습에 직접 적용해보기 

IoU를 로스로 사용해서 학습했을 때 CrossEntropy 를 사용한 것보다 mAP가 3.42% 향상되었다

$$L_{IoU} = 1- IoU = 1- \frac{TP}{FP+TP+FN}
$$


> Optimizing Intersection-Over-Union in Deep Neural Networks for Image Segmentation, 2016: [논문](http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf), [설명(K)](http://blog.naver.com/sogangori/221009464294)

