# mAP
- We use a metric called “mean average precision” (mAP)

- Compute average precision (AP) separately for each class, then average over classes

- A detection is a true positive if it has IoU with a ground-truth box greater than some threshold (usually 0.5) (mAP@0.5)

- Combine all detections from all test images to draw a precision / recall curve for each class; AP is area under the curve

- TL;DR mAP is a number from 0 to 100; high is good




  Mean Average Precision, 
- To calculate it for Object Detection, you calculate the average precision for each class in your data based on your model predictions. Average precision is related to the area under the precision-recall curve for a class. Then Taking the mean of these average individual-class-precision gives you the Mean Average Precision.

[[The PASCAL Visual Object Classes (VOC) Challenge]](http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf)

---
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

$$
L_{IoU} = 1- IoU = 1- \frac{TP}{FP+TP+FN}
$$


> Optimizing Intersection-Over-Union in Deep Neural Networks for Image Segmentation, 2016: [논문](http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf), [설명(K)](http://blog.naver.com/sogangori/221009464294)

## 3. 코드 

> 코드에 대한 상세 설명 : [Ronny Restrepo's Intersect Of Union (IoU)](http://ronny.rest/tutorials/module/localization_001/intersect_of_union/#)

```python
def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou
```


## 4. 3D IOU

[IoU설명](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)


```python 


def IoU(box0, box1):
  # box0: [x, y, z, d]
  r0 = box0[3] / 2
  s0 = box0[:3] - r0
  e0 = box0[:3] + r0
  r1 = box1[3] / 2
  s1 = box1[:3] - r1
  e1 = box1[:3] + r1
  
  overlap = [max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])) for i in range(3)]
  intersection = reduce(lambda x,y:x*y, overlap)
  union = pow(box0[3], 3) + pow(box1[3], 3) - intersection
  return intersection / union
```

> https://gist.github.com/daisenryaku/91e6f6d78f49f67602d21dc57d494c60
