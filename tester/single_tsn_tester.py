from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config
import sys
sys.path.append('/AccidentFaultAI/')
from module.accidentSearch import AccidentSearch

# 설정 파일을 선택하고 인식기를 초기화합니다.
config = '/AccidentFaultAI/model/TSN/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
config = Config.fromfile(config)

# 로드할 체크포인트 파일을 설정합니다.
checkpoint = '/AccidentFaultAI/model/TSN/best_model_0522.pth'

# 인식기를 초기화합니다.
model = init_recognizer(config, checkpoint, device='cuda:0')

# 인식기를 사용하여 추론을 수행합니다.
from operator import itemgetter

test_count = 0
rate_count = 0
total_count = 0
with open("/AccidentFaultAI/datasets/data/video_datasets/download_datas/custom_test_mp4.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()
    total_count = len(lines)

    for line in lines:
        video_name, video_label = line.split()

        # 예측할 비디오 파일 경로
        video = '/AccidentFaultAI/datasets/data/video_datasets/download_datas/test/'+video_name
        # 라벨 파일 경로
        label = '/AccidentFaultAI/model/TSN/index_map.txt'

        # 비디오에 대한 인식 결과를 얻습니다.
        results = inference_recognizer(model, video)

        # 예측 점수를 리스트로 변환합니다.
        pred_scores = results.pred_score.tolist()
        # 예측 점수와 인덱스를 튜플로 묶습니다.
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        # 점수를 기준으로 내림차순 정렬합니다.
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
        # 상위 5개의 라벨을 선택합니다.
        top5_label = score_sorted[:5]

        # 라벨 파일을 읽어옵니다.
        labels = open(label).readlines()
        # 라벨에서 공백 문자를 제거합니다.
        labels = [x.strip() for x in labels]

        # 상위 5개 라벨과 점수를 매핑합니다.
        results = [(labels[k[0]], k[1]) for k in top5_label]
        
        acs = AccidentSearch()
        
        result_type_dict = acs.select_type_num(int(video_label))[0]
        top_1_type_dict = acs.select_type_num(int(results[0][0]))[0]

        # # Debugging prints to check the structure
        # print(f"result_type_dict: {result_type_dict}")
        # print(f"top_1_type_dict: {top_1_type_dict}")

        # 상위 1개 가져오기
        print("정답 :"+video_label,end="")
        print(" | 비율{}:{}".format(result_type_dict["과실비율A"],result_type_dict["과실비율B"]))
        print(f'{results[0][0]}: ', results[0][1])
        print("{}:{}".format(top_1_type_dict["과실비율A"],top_1_type_dict["과실비율B"]))
        
        if int(results[0][0]) == int(video_label):
            test_count += 1
        if int(result_type_dict["과실비율A"]) == int(top_1_type_dict["과실비율A"]):
            rate_count += 1
            
print("top_1 정확도 {}|{} - {}%".format(test_count,total_count,test_count/total_count*100))
print("rate 정확도 {}|{} - {}%".format(rate_count,total_count,rate_count/total_count*100))

with open("/AccidentFaultAI/tester/single_tsn_tester_log.txt","a") as f:
    f.write("top_1 정확도 {}|{} - {}%\n".format(test_count,total_count,test_count/total_count*100))
    f.write("rate 정확도 {}|{} - {}%\n\n".format(rate_count,total_count,rate_count/total_count*100))