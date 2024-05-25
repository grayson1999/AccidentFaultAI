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
* [TSN (Temporal Segment Networks)](./video-swin-transformer.md)


## 목차
1. [데이터 설명](#데이터-설명) 
2. [환경설정](#환경설정)
3. [Version Control](#version-control)
4. [참고자료](#참고자료)

## 데이터 설명

- [AI-Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=597)
    
    - 데이터 설명서
    <br>[1-56 교통사고 영상 데이터_데이터설명서_v1.0.pdf](./asset/1-56%20교통사고%20영상%20데이터_데이터설명서_v1.0.pdf)
    - 데이터 사고유형별 index
    <br>[Incident_Type_Classification_Table.csv](./asset/Incident_Type_Classification_Table.csv)

## 환경설정
- docker 빌드
    ```bash
    # build an image with PyTorch 1.6.0, CUDA 10.1, CUDNN 7.
    docker build -f ./docker/Dockerfile --rm -t accidentfaultai .

    # docker run --gpus all --shm-size=8g -it accidentfaultai
    docker run --gpus all --shm-size=8g -it -v G:/:/accidentfaultai/datasets/data accidentfaultai

    ```
    ```bash
    #additional comments
    pip install mmcv==2.1.0
    pip install -r requirements/build.txt
    python setup.py develop
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


## 참고자료

- Object Detection
    
    [yolo v8 car crash detection | road accident detection yolo v8 | car crash detection project](https://www.youtube.com/watch?v=Hk2lGL1_EEg&t=263s)
    
    [GitHub - freedomwebtech/yolov8-vehicle-crash-detection](https://github.com/freedomwebtech/yolov8-vehicle-crash-detection/tree/main)
    
    [GitHub - shubhankar-shandilya-india/Accident-Detection-Model: Accident Detection Model using Deep Learning, OpenCV, Machine Learning, Artificial Intelligence.](https://github.com/shubhankar-shandilya-india/Accident-Detection-Model/tree/master)
    
    [YOLOv8 커스텀 데이터 학습하기](https://www.youtube.com/watch?v=em_lOAp8DJE)
    
    [YOLO v5 커스텀 학습 튜토리얼](https://www.youtube.com/watch?v=T0DO1C8uYP8)