from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config
import json
import sys
sys.path.append('/AccidentFaultAI/')
from module.accidentSearch import AccidentSearch
sys.path.append('/AccidentFaultAI/yolov8')
from module.yolov8.yolo_predict import Yolo_predect

# 인식기를 사용하여 추론을 수행합니다.
from operator import itemgetter

class Yolo_tsn_recognizer():
    def __init__(self):
        # 설정 파일을 선택하고 인식기를 초기화합니다.
        config = '/AccidentFaultAI/model/TSN/best_model_0527/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
        config = Config.fromfile(config)
        
        # 로드할 체크포인트 파일을 설정합니다.
        checkpoint = '/AccidentFaultAI/model/TSN/best_model_0527/best_model_0527.pth'

        # 인식기를 초기화합니다.
        self.model = init_recognizer(config, checkpoint, device='cuda:0')

        # yolo 인식기 초기화
        self.ypd = Yolo_predect()
    
    def predict(self, video_path):
        # 예측할 비디오 파일 경로
        video = video_path
        # 라벨 파일 경로
        label = '/AccidentFaultAI/model/index_map.txt'
        
        # 비디오에 대한 인식 결과를 얻습니다.
        results = inference_recognizer(self.model, video)
        yolo_result_dict = self.ypd.predict(video)
        
        # 예측 점수를 리스트로 변환합니다.
        pred_scores = results.pred_score.tolist()
        # 예측 점수와 인덱스를 튜플로 묶습니다.
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        # 점수를 기준으로 내림차순 정렬합니다.
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
        # 상위 10개의 라벨을 선택합니다.
        top10_label = score_sorted[:10]
        
        # 라벨 파일을 읽어옵니다.
        labels = open(label).readlines()
        # 라벨에서 공백 문자를 제거합니다.
        labels = [x.strip() for x in labels]
        
        # 상위 10개 라벨과 점수를 매핑합니다.
        results = [(labels[k[0]], k[1]) for k in top10_label]
        
        # 필터링 한 AccidentType category list 
        acs = AccidentSearch()
        filter_category = self.ypd.filter_detection(yolo_result_dict)
        
        ##필터 된 result 중 상위 1개 가져오기
        new_top5_results = []
        new_result = []
        
        for result in results:
            new_result.append((result[0],result[1]))
            if len(new_top5_results)>1:
                break
            
            if result[0] in filter_category:
                new_top5_results.append((result[0],result[1]))
        
        ## 모든 객체가 필터링 되어 empty 일 경우 예외 처리
        if len(new_top5_results) == 0:
            new_top5_results == new_result[:1]
            
        result_type_dict = acs.select_type_num(int(new_top5_results[0][0]))[0]
        # 딕셔너리를 JSON 파일로 저장
        with open('/AccidentFaultAI/recognizer/output/yolo_tsn_result.json', 'w', encoding='utf-8') as file:
            json.dump(result_type_dict, file, ensure_ascii=False, indent=4)

        return result_type_dict
    

if __name__=="__main__":
    ysr = Yolo_tsn_recognizer()
    print(ysr.yolo_tsn_predict("./demo_video/cc_5.mp4"))