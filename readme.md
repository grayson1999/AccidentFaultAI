# AI-based traffic accident negligence measurement program

## 프로젝트의 목적

- 주어진 비디오를 입력으로 받아들여 객체 감지를 수행하고, 이를 기반으로 사전에 정의된 435가지 상황 데이터 중에서 가장 유사한 상황을 예측하여 과실 비율을 측정
    
    ```mermaid
    graph LR
    nID1[video];
    nID2[YOLOv8\ndetection];
    nID3[Temporal Segment Networks\nrecognizor];
    nID4[text\nnegligence rate];
    nID1--mp4-->nID2--dict-->nID3--set-->nID4
    ```

* [YOLOv8 detection 모델]()
* [TSN (Temporal Segment Networks)](https://github.com/grayson1999/TSNAccidentAnalysis)


## 목차
1. [데이터 설명](#데이터-설명) 
2. [환경설정](#환경설정)
3. [모델 정확도](#모델-정확도)
4. [Single_TSN_model](#single_tsn_model)
5. [Version Control](#version-control)
6. [참고자료](#참고자료)

## 데이터 설명

- [AI-Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=597)
    
    - 데이터 설명서
    <br>[1-56 교통사고 영상 데이터_데이터설명서_v1.0.pdf](./asset/1-56%20교통사고%20영상%20데이터_데이터설명서_v1.0.pdf)
    - 데이터 사고유형별 index
    <br>[Incident_Type_Classification_Table.csv](./files/Incident_Type_Classification_Table.csv)

## 환경설정
- docker 빌드
    ```bash
    # 버전 수정
    ARG PYTORCH="1.6.0"
    ARG CUDA="10.1"
    ARG CUDNN="7"
    ```
    ```bash
    # build an image with PyTorch 1.6.0, CUDA 10.1, CUDNN 7.
    docker build -f ./docker/Dockerfile --rm -t accidentfaultai .

    # docker run --gpus all --shm-size=8g -it accidentfaultai
    docker run --gpus all --shm-size=8g -it -v G:/:/AccidentFaultAI/datasets/data accidentfaultai

    ```
    ```bash
    #additional comments
    pip install mmcv==2.1.0
    pip install -r requirements/build.txt
    python setup.py develop
    pip install pandas
    ```

## 모델 정확도
|     모듈     |      모델 설명          |  top_1 정확도    |     top_5 정확도    |     rate 정확도    |
|--------------|-----------------------|------------------|---------------------|-------------------|
|single_tsn_model|TSN(best_model_0522)|20.6|·|29.9|
|single_tsn_model|TSN(best_model_0527)|23.0|46.8|32.0|

## Single_TSN_model
- 경로 수정
    ```python
    ## ./recognizer/single_tsn_recognizer.py
    config = '/AccidentFaultAI/model/TSN/best_model_0527/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
    checkpoint = '/AccidentFaultAI/model/TSN/best_model_0527/best_model_0527.pth'
    video = '/AccidentFaultAI/recognizer/demo_video/cc_5.mp4'
    ```
- run   
    ```bash
    python ./recognizer/single_tsn_recognizer.py
    ```

## Version Control

| 버전       | 날짜      | 변경 내용                                |
|------------|-------------|------------------------------------------|
|0.1|24.05.22|초기 repository 설정 및 초기 video-swin-transformer 버전 업로드|
|0.2|24.05.22|main, video-swin-transformer의 readme.md 작성|
|0.3|24.05.22|사고 유형을 통해 과실 비율 등 검색 모듈 추가(accidentSerch)|
|0.3.1|24.05.24|main docker file 및 model 업로드|
|0.3.2|24.05.24|main docker file 수정, single_tsn_teater 제작, main readme 환경 설정 섹션 추가|
|0.3.3|24.05.25|video-swin-transformer를 main 폴더로 정리|  
|0.3.4|24.05.27|video-swin-transformer를 [TSNAccidentAnalysis](https://github.com/grayson1999/TSNAccidentAnalysis)으로 분리, docker file 수정|  
|0.4|24.05.27|docker 환경 구축, single_tsn_model 테스트| 
|0.4.1|24.05.27|top_5 acc 추가, 모델(best_model_0527) 추가|
|0.5|24.05.27|recognizer 제작|
|0.5.1|24.05.27|incident_Type 컬럼 명 영어로 변경 및 모듈 대응|
|0.5.2|24.05.27|yolo 환경 설정 추가|


## 참고자료

- Object Detection
    
    [yolo v8 car crash detection | road accident detection yolo v8 | car crash detection project](https://www.youtube.com/watch?v=Hk2lGL1_EEg&t=263s)
    
    [GitHub - freedomwebtech/yolov8-vehicle-crash-detection](https://github.com/freedomwebtech/yolov8-vehicle-crash-detection/tree/main)
    
    [GitHub - shubhankar-shandilya-india/Accident-Detection-Model: Accident Detection Model using Deep Learning, OpenCV, Machine Learning, Artificial Intelligence.](https://github.com/shubhankar-shandilya-india/Accident-Detection-Model/tree/master)
    
    [YOLOv8 커스텀 데이터 학습하기](https://www.youtube.com/watch?v=em_lOAp8DJE)
    
    [YOLO v5 커스텀 학습 튜토리얼](https://www.youtube.com/watch?v=T0DO1C8uYP8)