import os
import json
import random
import shutil

##dictionary 형식을 txt파일 형식으로 변환 저장
def save_dict_to_txt(dictionary, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for key, value in dictionary.items():
            file.write(f'{key} {value}\n')
            
# 디렉토리 생성 함수 정의
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"디렉토리 '{directory_path}'를 생성했습니다.")
    else:
        print(f"디렉토리 '{directory_path}'는 이미 존재합니다.")

def copy_files_from_list(file_list_path, destination_path):
    # 대상 폴더가 존재하지 않으면 생성
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # 파일 열고 각 줄을 리스트로 읽기
    with open(file_list_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 각 줄에서 원본 파일 경로를 읽고 파일을 복사
    for line in lines:
        origin_file_path = line.split()[0]  # 각 줄의 첫 번째 항목을 원본 파일 경로로 사용
        destination_file_path = os.path.join(destination_path, os.path.basename(origin_file_path))
        
        # 파일 복사
        shutil.copyfile(origin_file_path, destination_file_path)
        print(f"파일 '{origin_file_path}'을(를) '{destination_file_path}'로 복사했습니다.")



# 입력 파일의 각 줄에서 파일 이름과 라벨을 추출하여 출력 파일에 저장
def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_file, 'w', encoding='utf-8') as file:
        for line in lines:
            parts = line.strip().split()
            mp4_file_with_path = parts[0]
            label = parts[1]
            mp4_file = mp4_file_with_path.split('/')[-1]  # 경로에서 파일 이름만 추출
            file.write(f"{mp4_file} {label}\n")

    print(f"파일 '{output_file}'에 mp4 파일명과 라벨이 저장되었습니다.")
    
## 비디오 어노테이션 결과를 저장할 딕셔너리
result_dict = {}

# 레이블 및 비디오 폴더 경로 설정
label_folder_path = "./label/"
video_folder_path = "./origin/"

# 레이블 폴더 내의 폴더 목록 가져오기
folder_names = list(map(lambda x:"_".join(x.split("_")[1:]), os.listdir(label_folder_path)))

# 각 폴더별로 작업 수행
for folder_name in folder_names:
    current_folder_path = label_folder_path + "TL_" + folder_name + "/"
    current_folder_labels = os.listdir(current_folder_path)
    
    # 폴더 내의 각 레이블에 대해 처리
    for current_label in current_folder_labels:
        with open(current_folder_path + current_label) as f:
            data = json.load(f)["video"]
        
        # 레이블 데이터에서 비디오 경로 및 정답 값 추출하여 딕셔너리에 저장
        try:
            if data["traffic_accident_type"] < 434:
                result_dict[video_folder_path + "TS_" + folder_name + "/" + data["video_name"]+".mp4"] = data["traffic_accident_type"]
        except KeyError:
            try:
                if data["accident_type"] < 434:
                    result_dict[video_folder_path + "TS_" + folder_name + "/" + data["video_name"]+".mp4"] = data["accident_type"]
            except KeyError:
                continue

# 데이터를 무작위로 섞음
video_label_pairs = list(result_dict.items())
random.shuffle(video_label_pairs)

# 데이터 비율 설정
total_samples = len(video_label_pairs)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 데이터를 비율에 맞게 분할
train_samples = int(total_samples * train_ratio)
val_samples = int(total_samples * val_ratio)
test_samples = total_samples - train_samples - val_samples

# 각 세트에 속하는 데이터 추출
train_set = video_label_pairs[:train_samples]
val_set = video_label_pairs[train_samples:train_samples + val_samples]
test_set = video_label_pairs[train_samples + val_samples:]

# 결과를 파일에 저장
save_dict_to_txt(dict(train_set), 'custom_train.txt')
save_dict_to_txt(dict(val_set), 'custom_val.txt')
save_dict_to_txt(dict(test_set), 'custom_test.txt')

# 파일 경로와 이동할 폴더 경로 지정
file_list_paths = ["custom_train.txt", "custom_val.txt", "custom_test.txt"]
destination_paths = ["./train", "./val", "./test"]

# 각 파일에 대해 파일 이동 작업 수행
for file_list_path, destination_path in zip(file_list_paths, destination_paths):
    copy_files_from_list(file_list_path, destination_path)
    
# 각 파일에 대해 annotation 수정 작업 수행
files = ['custom_train.txt', 'custom_val.txt', 'custom_test.txt']

for file_name in files:
    input_file = file_name
    output_file = file_name.replace('.txt', '_mp4.txt')
    process_file(input_file, output_file)