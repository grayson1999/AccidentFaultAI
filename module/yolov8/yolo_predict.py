import cv2
from ultralytics import YOLO
import pandas as pd
import sys
sys.path.append('/AccidentFaultAI/')
from module.accidentSearch import AccidentSearch

class Yolo_predect():
    def __init__(self):
        # YOLO 모델 로드
        self.model = YOLO("yolov8n.pt")
        # 예측할 클래스 인덱스 설정
        self.target_classes = [0, 1, 2, 3, 5, 7, 9, 11]
        # 클래스 이름과 인덱스 매핑
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign'
        }
        # 클래스별 객체 수를 저장할 딕셔너리 초기화
        self.class_counts = {
            'people': 0, 'bicycles': 0, 'motorcycles': 0, 
            'vehicles': 0, 'traffic lights': 0, 'sign': 0
        }
        self.detected_objects = set()


    def predict(self,video_path):
        # 비디오 파일 열기
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 각 프레임에서 객체 탐지
            results = self.model.predict(source=frame, classes=self.target_classes,verbose=False)
            boxes = results[0].boxes

            current_frame_objects = set()
            for box in boxes:
                class_id = int(box.cls)
                class_name = self.class_names.get(class_id)
                if class_name:
                    # Bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    object_id = (class_name, x1, y1, x2, y2)
                    current_frame_objects.add(object_id)

                    if object_id not in self.detected_objects:
                        self.detected_objects.add(object_id)
                        if class_name == 'person':
                            self.class_counts['people'] += 1
                        elif class_name == 'bicycle':
                            self.class_counts['bicycles'] += 1
                        elif class_name == 'car' or class_name == 'bus' or class_name == 'truck':
                            self.class_counts['vehicles'] += 1
                        elif class_name == 'motorcycle':
                            self.class_counts['motorcycles'] += 1
                        elif class_name == 'traffic light':
                            self.class_counts['traffic lights'] += 1
                        elif class_name == 'stop sign':
                            self.class_counts['sign'] += 1

            # 중복 방지를 위해 현재 프레임에서 감지된 객체만 유지
            self.detected_objects = self.detected_objects.intersection(current_frame_objects)

            # 결과 출력 (프레임별로 보여주기 위해 주석 처리)
            # for obj in current_frame_objects:
            #     class_name, x1, y1, x2, y2 = obj
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.imshow('Frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cap.release()
        cv2.destroyAllWindows()

        # 최종 클래스별 객체 수 출력
        return self.class_counts
    
    def filter_detection(self,detection_dict):

        ads = AccidentSearch()
        data = ads.get_all_data()

        ## 필터링 할 데이터
        result = pd.DataFrame()

        ## filtering
        if detection_dict["motorcycles"] == 0:
            temp_df = data.loc[data["AccidentObject"] == "차대이륜차"]
            result = pd.concat([result,temp_df],ignore_index = True)
        if detection_dict["people"] == 0:
            temp_df = data.loc[data["AccidentObject"] == "차대보행자"]
            result = pd.concat([result,temp_df],ignore_index = True)
        if detection_dict["bicycles"] == 0:
            temp_df = data.loc[data["AccidentObject"] == "차대자전거"]
            result = pd.concat([result,temp_df],ignore_index = True)

        ## 특수 조건 filtering
        if detection_dict["traffic lights"] == 0:
            temp_df = data.loc[data["AccidentLocation"].str.contains("신호등 있음")]
            result = pd.concat([result,temp_df],ignore_index = True)
        if detection_dict["traffic lights"] > 0:
            temp_df = data.loc[data["AccidentLocation"].str.contains("신호등 없음")]
            result = pd.concat([result,temp_df],ignore_index = True)

        
        ##filter result 중복제거 
        filter_result = result.drop_duplicates()

        ##차집합으로 data filter
        data = pd.merge(data, filter_result, how="outer", indicator = True).query('_merge == "left_only"').drop(columns=['_merge'])
        filter_category = list(map(str,list(data["AccidentType"])))
        return filter_category
        