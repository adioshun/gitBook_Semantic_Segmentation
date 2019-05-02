# [Introduction to Motion Estimation with Optical Flow](https://blog.nanonets.com/optical-flow/)


대부분의 컴퓨터 비젼은 시간의 흐름에 상관 없이 각 프레임의 정보만을 고려 한다. 만일 각 프레임 마다 연관성이 있고 이를 고려 한다면 어떤 기술이 가능할까? `only address relationships of objects within the same frame (x,y) disregarding time information (t). In other words, they re-evaluate each frame independently, as if they are completely unrelated images, for each run. However, what if we do need the relationships between consecutive frames, for example, we want to track the motion of vehicles across frames to estimate its current velocity and predict its position in the next frame?`
   

## 1. 개요 
 
  
    
Optical flow is the motion of objects between consecutive frames of sequence, caused by the relative movement between the object and camera.

