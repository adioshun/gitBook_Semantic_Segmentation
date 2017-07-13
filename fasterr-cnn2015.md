|논문명|Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks|
|-|-|
|학회/년도|NIPS 2015, [논문](https://arxiv.org/pdf/1506.01497.pdf),[코드](https://github.com/rbgirshick/py-faster-rcnn)|
|참고|[PR-012](https://www.youtube.com/watch?v=kcPAGIgBGRs&feature=youtu.be&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS)|


# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

## 0. Abstract 

Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals
`region proposals을 위해 PRN은 Detection Netowkr와 features 를 공유 한다.`

We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features—using the recently popular terminology of neural networks with “attention” mechanisms, the RPN component tells the unified network where to look
`최종적으로 “attention” mechanisms을 이용하여features을 공유 함으로써  RPN와 Fast R-CNN를 하나의 네트워크로 합칠수 있다.`

## 1. Introduction 

오늘날의 물체 탐지 기술은 region proposal methods (e.g., [4]) & region-based convolutional neural networks (RCNNs) [5]주도로 발전해 왔다. 

Fast R-CNN [2], achieves near real-time rates using very deep networks [3], when ignoring the time spent on region proposals. Region proposal methods typically rely on inexpensive features and economical inference schemes.
> R-CNN이 속도 개선을 하였지만, **region proposals**시간을 고려 하지 않았을때만 속도 개선이 있다. Proposal은 아직도 시간 소모가 크다.
> Region proposal는 원래 시간이 오래 걸리지 않는 추정 방식이다.`

Selective Search [4], one of the most popular methods, greedily merges superpixels based on engineered low-level features. Yet when compared to efficient detection networks [2], Selective Search is an order of magnitude slower, at 2 seconds per image in a CPU implementation. EdgeBoxes [6] currently provides the best tradeoff between proposal quality and speed,at 0.2 seconds per image. Nevertheless, the region proposal step still consumes as much running time as the detection network.
>가장 많이 사용하는 방법인 Selective Search 방법도 초당 2초 정도 소모될정도로 느리다. 최선의 EdgeBoxes도 0.2초가 걸린다. 그럼에도 불구 하고 region proposal단계는 Detection Network에서 가장 많은 시간을 잡아 먹는다.

One may note that fast region-based CNNs takeadvantage of GPUs, while the region proposal methods used in research are implemented on the CPU,making such runtime comparisons inequitable. An obvious way to accelerate proposal computation is to re implement it for the GPU. This may be an effective engineering solution, but re-implementation ignores the down-stream detection network and therefore misses important opportunities for sharing computation. 
> F-CNN은 GPU사용의 장점도 가지고 있다. 공정한 테스트를 위해서는 CPU에서 테스트된 region proposal방법도 GPU용으로 재 구혀 하여야 한다. 하지만, 재구현은 "down-stream detection network"을 무시하게 되고 이로 인해 sharing computation으로 인한  important opportunities를 놓치게 된다. (???)

In this paper, we show that an algorithmic change—computing proposals with a deep convolutional neural network—leads to an elegant and effective solution where proposal computation is nearly cost-free given the detection network’s computation. To this end, we introduce novel Region Proposal Networks (RPNs) that share convolutional layers with state-of-the-art object detection networks [1], [2]. By sharing convolutions at test-time, the marginal cost for computing proposals is small (e.g., 10ms per image).
> 본 논문에서는 [CNN으로 proposals을 계산]한는 챌리지를 개선 하였다. 제안된 Region Proposal Networks (RPNs)는 convolutional layers를 기존의 object detection networks와 공유 한다. convolutions을 테스트시 공유 함으로써 proposals 계산 시간을 단축 하였다. 

Our observation is that the convolutional feature maps used by region-based detectors, like Fast RCNN, can also be used for generating region proposals. On top of these convolutional features, we construct an RPN by adding a few additional convolutional layers that simultaneously regress region bounds and objectness scores at each location on a regular grid. The RPN is thus a kind of fully convolutional network (FCN) [7] and can be trained end-to-end specifically for the task for generating detection proposals.
> 우리의 통찰 결과 Fast RCNN에서 사용되는 [convolutional feature maps]은 region proposals을 생성할때도 사용 될수 있다는 점이다. 이 [convolutional features]위에 RPN를 추가 하고 
> - RPN : Convolutional layers that simultaneously regress region bounds and objectness scores at each location on a regular grid.
> - RPN은 일종의 fully convolutional network (FCN) [7]이다. (논문 7번 읽어 보기)

![](http://i.imgur.com/AeQXiE8.png)

RPNs are designed to efficiently predict region proposals with a wide range of scales and aspect ratios. In contrast to prevalent methods [8], [9], [1], [2] that use pyramids of images (Figure 1, a) or pyramids of filters(Figure 1, b), we introduce novel “anchor” boxes that serve as references at multiple scales and aspect ratios. Our scheme can be thought of as a pyramid of regression references (Figure 1, c), which avoids enumerating images or filters of multiple scales or aspect ratios. This model performs well when trained and tested using single-scale images and thus benefits running speed 
> 기존 방식은 pyramids of images (Figure 1, a) 나 pyramids of filters(Figure 1, b)를 사용했지만, RPN은 “anchor”박스를 제안 한다. 
> - anchors는 serve as references at multiple scales and aspect ratios

> 우리 방식은 pyramid of regression references로 볼수 있으며, 많은 수/크기의 이미지나 필터들이 불필요 하다. 한 크기(single-scale)의 이미지만 사용하여서 속도가 빠른 것이다. 

To unify RPNs with Fast R-CNN [2] object detection networks, we propose a training scheme that alternates between fine-tuning for the region proposal task and then fine-tuning for object detection, while keeping the proposals fixed. This scheme converges quickly and produces a unified network with convolutional features that are shared between both tasks.
> RPNs과  Fast R-CNN object detection networks를 하나로 합치기 위해서 region proposal을 위한 파인 튜닝과 object detection를 위한 파인튜닝을 번갈아 가면서 수행하는 방법을 제안 한다. 이 방법을 통해 convolutional features(CF)로 된 통일된 네트워크가 만들어 진다. CF는 위 두 튜닝과정이 벌갈아 수행될떄 서로 공유 된다. 

We comprehensively evaluate our method on thePASCAL VOC detection benchmarks [11] where RPNswith Fast R-CNNs produce detection accuracy better than the strong baseline of Selective Search withFast R-CNNs. Meanwhile, our method waives nearlyall computational burdens of Selective Search attest-time—the effective running time for proposalsis just 10 milliseconds. Using the expensive verydeep models of [3], our detection method still hasa frame rate of 5fps (including all steps) on a GPU,and thus is a practical object detection system interms of both speed and accuracy. We also reportresults on the MS COCO dataset [12] and investigate the improvements on PASCAL VOC using theCOCO data. Code has been made publicly availableat https://github.com/shaoqingren/faster_rcnn (in MATLAB) and https://github.com/rbgirshick/py-faster-rcnn (in Python).
> 성능 평가 결과도 좋다. 코드는 MATLAB과  Python으로 작성되어 공개 하였다. 

