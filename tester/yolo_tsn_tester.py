from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config
import sys
sys.path.append('/AccidentFaultAI/')
from module.accidentSearch import AccidentSearch
sys.path.append('/AccidentFaultAI/yolov8')
from module.yolov8.yolo_predict import Yolo_predect


# 설정 파일을 선택하고 인식기를 초기화합니다.
config = '/AccidentFaultAI/model/TSN/best_model_0527/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
config = Config.fromfile(config)

# 로드할 체크포인트 파일을 설정합니다.
checkpoint = '/AccidentFaultAI/model/TSN/best_model_0527/best_model_0527.pth'

# 인식기를 초기화합니다.
model = init_recognizer(config, checkpoint, device='cuda:0')

ypd = Yolo_predect()

# 인식기를 사용하여 추론을 수행합니다.
from operator import itemgetter

count = 0
test_count = 0
test5_count = 0
rate_count = 0
total_count = 0
with open("/AccidentFaultAI/datasets/data/video_datasets/download_datas/custom_test_mp4.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()
    total_count = len(lines)

    for line in lines:
        count+=1
        video_name, video_label = line.split()

        # 예측할 비디오 파일 경로
        video = '/AccidentFaultAI/datasets/data/video_datasets/download_datas/test/'+video_name
        # 라벨 파일 경로
        label = '/AccidentFaultAI/model/index_map.txt'

        # 비디오에 대한 인식 결과를 얻습니다.
        results = inference_recognizer(model, video)
        yolo_result_dict = ypd.predict(video)
        

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
        filter_category = ypd.filter_detection(yolo_result_dict)
        
        ##필터 된 result 중 상위 5개 가져오기
        new_top5_results = []
        new_result = []
        for result in results:
            new_result.append((result[0],result[1]))
            if len(new_top5_results)>5:
                break
            if result[0] in filter_category:
                new_top5_results.append((result[0],result[1]))

        ## 모든 객체가 필터링 되어 empty 일 경우 예외 처리
        if len(new_top5_results) == 0:
            new_top5_results == new_result[:5]

        result_type_dict = acs.select_type_num(int(video_label))[0]
        top_1_type_dict = acs.select_type_num(int(new_top5_results[0][0]))[0]

        # # Debugging prints to check the structure
        # print(f"상위 10개 기본: {results}")
        # print(f"필터링 한 상위 5개: {new_top5_results}")
        # print(f"필터 list: {filter_category}")
        # print(f"result_type_dict: {result_type_dict}")
        # print(f"top_1_type_dict: {top_1_type_dict}")

        # # 상위 1개 가져오기
        # print("정답 :"+video_label,end="")
        # print(" | 비율 {}:{}".format(result_type_dict["FaultRatioA"],result_type_dict["FaultRatioB"]))
        # print("{}:{}".format(top_1_type_dict["FaultRatioA"],top_1_type_dict["FaultRatioB"]))

        for result in new_top5_results:
            # print(f'{result[0]}: ', result[1])
            if int(video_label) == int(result[0]):
                test5_count += 1
        
        
        if int(new_top5_results[0][0]) == int(video_label):
            test_count += 1

        if int(result_type_dict["FaultRatioA"]) == int(top_1_type_dict["FaultRatioA"]):
            rate_count += 1

        #debug
        # print(test_count)
        # print(test5_count)
        # print(rate_count)
        # print(total_count)

        print("-"*3+"진행률 {}/{}".format(count,total_count)+"-"*3)  
        print("top_1 정확도 {}|{} - {}%".format(test_count,total_count,test_count/count*100))
        print("top_5 정확도 {}|{} - {}%".format(test5_count,total_count,test5_count/count*100))
        print("rate 정확도 {}|{} - {}%".format(rate_count,total_count,rate_count/count*100))

with open("/AccidentFaultAI/tester/yolo_tsn_log.txt","a") as f:
    f.write("top_1 정확도 {}|{} - {}%\n".format(test_count,total_count,test_count/total_count*100))
    f.write("top_5 정확도 {}|{} - {}%\n".format(test5_count,total_count,test5_count/total_count*100))
    f.write("rate 정확도 {}|{} - {}%\n\n".format(rate_count,total_count,rate_count/total_count*100))