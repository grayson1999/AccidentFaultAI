import os
import cv2

def is_video_file_corrupted(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return True
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return True
        
        cap.release()
    except Exception as e:
        print(f"Exception while checking video: {e}")
        return True

    return False

def filter_corrupted_videos(annotation_file,video_root, output_file):
    error_videos = []

    with open(annotation_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            filename, _ = parts
            video_path = os.path.join(video_root, filename)

            if is_video_file_corrupted(video_path):
                error_videos.append(video_path)

    with open(output_file, 'w') as file:
        for video in error_videos:
            file.write(video + '\n')

    print(f"Error videos have been saved to {output_file}")

def main():
    annotation_file = os.path.join('./custom_val_mp4.txt')
    output_file = os.path.join('./file_check_log.txt')

    filter_corrupted_videos(annotation_file,'./val', output_file)

if __name__ == "__main__":
    main()