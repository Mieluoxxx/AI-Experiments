from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import torch


def _display_detected_frames(conf, model, st_frame, image):
    """
    使用YOLOv8模型在视频帧上显示检测到的对象。
    :param conf (float): 对象检测的置信度阈值。
    :param model (YOLOv8): 包含YOLOv8模型的`YOLOv8`类的实例。
    :param st_frame (Streamlit对象): 用于显示检测到的视频的Streamlit对象。
    :param image (numpy数组): 表示视频帧的numpy数组。
    :return: 无
    """
    # 将图像调整为标准大小
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # 使用YOLOv8模型在图像中预测对象
    res = model.predict(image, conf=conf)

    # 在视频帧上绘制检测到的对象
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption="检测到的视频", channels="BGR", use_column_width=True)


@st.cache_resource
def load_model(model_path):
    """
    从指定的model_path加载YOLO目标检测模型。

    参数:
        model_path (str): YOLO模型文件的路径。

    返回:
        YOLO目标检测模型。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path).to(device)
    return model


def infer_uploaded_image(conf, model):
    """
    执行上传图像的推断
    :param conf: YOLOv8模型的置信度
    :param model: 包含YOLOv8模型的`YOLOv8`类的实例。
    :return: 无
    """
    source_img = st.sidebar.file_uploader(
        label="选择图像...", type=("jpg", "jpeg", "png", "bmp", "webp")
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # 将上传的图像添加到页面并添加标题
            st.image(image=source_img, caption="上传的图像", use_column_width=True)

    if source_img:
        if st.button("执行"):
            with st.spinner("运行中..."):
                res = model.predict(uploaded_image, conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted, caption="检测到的图像", use_column_width=True)
                    try:
                        with st.expander("检测结果"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("尚未上传图像！")
                        st.write(ex)


def infer_uploaded_video(conf, model):
    """
    执行上传视频的推断
    :param conf: YOLOv8模型的置信度
    :param model: 包含YOLOv8模型的`YOLOv8`类的实例。
    :return: 无
    """
    source_video = st.sidebar.file_uploader(label="选择视频...")

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("执行"):
            with st.spinner("运行中..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(tfile.name)
                    st_frame = st.empty()
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf, model, st_frame, image)
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"加载视频时出错：{e}")


def infer_uploaded_webcam(conf, model):
    """
    执行摄像头推断。
    :param conf: YOLOv8模型的置信度
    :param model: 包含YOLOv8模型的`YOLOv8`类的实例。
    :return: `无
    """
    try:
        flag = st.button(label="停止运行")
        vid_cap = cv2.VideoCapture(0)  # 本地摄像头
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf, model, st_frame, image)
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"加载视频时出错：{str(e)}")
