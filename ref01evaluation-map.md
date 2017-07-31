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

|![](http://i.imgur.com/HXXb6WX.png)|![](http://i.imgur.com/LjtA6h8.png)|
|-|-|

> 출처 : [Intersection over Union (IoU) for object detection](http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
