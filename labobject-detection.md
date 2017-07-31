# 실습 Object Detection 

## 1. Tensorflow Object Detection API

> [Anaconda(spyder)를 이용한 Tensorflow Object Detection API](http://boysboy3.tistory.com/98)





> [Tensorflow 를 이용한 Object Detection API 소개](http://eehoeskrap.tistory.com/156)

> [Tensorflow Object Detection API (SSD, Faster-R-CNN)](http://goodtogreate.tistory.com/entry/Tensorflow-Object-Detection-API-SSD-FasterRCNN)

> [How to train your own Object Detector with TensorFlow’s Object Detector API](https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)

---

Object Detection 의 사용 지침에 오류가 좀 있습니다. 
처음 api 나오고 나서 프로젝트는 계속 업데이트 되는데 readme 는
업데이트가 한번도 안됐네요ㅎㅎ tensorflow models 프로젝트의 
export_inference_graph.py 파일 실행시 필요한 parameter 의 이름이 바뀌었습니다. 
https://github.com/…/object_detec…/g3doc/exporting_models.md 에 들어가면 사용법이 나와있긴 하지만 몇가지 틀린 부분이 있어서, 실행 해도 오류가 납니다.
기존의 checkpoint_path 는 trained_checkpoint_prefix 로
inference_graph_path 는 output_directory 로 바뀌었습니다. 참고 하세요. 
한가지 더...
python object_detection/export_inference_graph \ 가 아니라
python object_detection/export_inference_graph.py \ 입니다.