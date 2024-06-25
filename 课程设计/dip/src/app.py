from pathlib import Path
from PIL import Image
import streamlit as st

import config
from utils import (
    load_model,
    infer_uploaded_image,
    infer_uploaded_video,
    infer_uploaded_webcam,
)

# 设置页面布局
st.set_page_config(
    page_title="YOLOv8交互界面",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 主页面标题
st.title("YOLOv8交互界面")

# 侧边栏
st.sidebar.header("深度学习模型配置")

# 模型选项
task_type = st.sidebar.selectbox("选择任务", ["检测"])

model_type = None
if task_type == "检测":
    model_type = st.sidebar.selectbox("选择模型", config.DETECTION_MODEL_LIST)
else:
    st.error("目前只实现了 '检测' 功能")

confidence = float(st.sidebar.slider("选择模型置信度", 30, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("请在侧边栏中选择模型")

# 加载预训练的深度学习模型
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"无法加载模型。请检查指定的路径：{model_path}")

# 图像/视频选项
st.sidebar.header("图像/视频配置")
source_selectbox = st.sidebar.selectbox("选择来源", config.SOURCES_LIST)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]:  # 图像
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]:  # 视频
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]:  # 摄像头
    infer_uploaded_webcam(confidence, model)
else:
    st.error("目前只实现了 '图像' 和 '视频' 来源")
