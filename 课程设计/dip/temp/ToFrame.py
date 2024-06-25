import cv2
import os

def video_to_frames(video_path, output_path, frame_interval=20):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 确保视频文件已成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频的帧率和帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}")
    print(f"Total frames: {frame_count}")

    # 初始化计数器
    frame_number = 0

    # 逐帧读取视频并保存为图像文件
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_number}")
            break

        # 每隔 frame_interval 帧保存一次图像
        if frame_number % frame_interval == 0:
            # 生成图像文件名
            frame_filename = f"{output_path}/frame_{frame_number:04d}.jpg"

            # 保存图像文件
            cv2.imwrite(frame_filename, frame)

        frame_number += 1

    # 释放视频对象
    cap.release()

if __name__ == "__main__":
    # 输入视频文件路径和输出帧图像的目录
    input_video_path = os.getcwd() + "/data/raw/supermario.mp4"
    output_frames_path = os.getcwd() + "/data/processed/"

    # 调用函数进行转换
    video_to_frames(input_video_path, output_frames_path)
