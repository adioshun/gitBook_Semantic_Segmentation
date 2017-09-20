![](https://cdn-images-1.medium.com/max/800/1*glryPZvHJzGa8SDXZ4YuXA.png)

> 참고 : [The Basics of Video Object Segmentation](https://medium.com/@eddiesmo/video-object-segmentation-the-basics-758e77321914)


# Dataset 

- DAVIS : It aims to recreate real-life video scenarios such as camera shake, background clutter, occlusions and other complexities.


###### DAVIS-2016 complexity attributes
![](https://cdn-images-1.medium.com/max/800/1*y5IuhAwd4Elznm9JRT9YVg.png)

- [GyGO](https://github.com/ilchemla/gygo-dataset) : about 150 short videos
    - The GyGO dataset specializes in smartphone captured videos and its frames are relatively sparse (the annotated video speed is ~5 fps).
    
### Goal of GyGo

- There is a severe lack of data in the space of video object segmentation at the moment. With only hundreds of annotated videos, we believe every contribution has the potential to increase performance. In our analysis we have shown that a joint training on the GyGO and DAVIS datasets yields improved inference results.

- To promote a more open, sharing culture and encourage other researchers to join our efforts :) The DAVIS dataset and the research ecosystem that grew it have been massively useful for us. We hope the community will find our datasets useful as well.


---
- Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung, Computer Vision and Pattern Recognition (CVPR) 2016

- The 2017 DAVIS Challenge on Video Object Segmentation
J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbeláez, A. Sorkine-Hornung, and L. Van Gool, arXiv:1704.00675, 2017