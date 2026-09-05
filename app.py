import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

st.set_page_config(page_title="YOLOv8 Detection", layout="wide")
st.title("🔍 YOLOv8 Object Detection")
st.write("Upload image for detection")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')
model = load_model()

uploaded_file = st.file_uploader("Choose image", type=["jpg","png","jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded", use_container_width=True)
    if st.button("Detect Objects"):
        with st.spinner("Detecting..."):
            results = model(np.array(image))
            plotted = results[0].plot()
            plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            st.image(plotted, caption="Result", use_container_width=True)
            for box in results[0].boxes:
                st.write(f"- {model.names[int(box.cls[0])]} : {float(box.conf[0]):.2f}")
else:
    st.info("Upload an image to start")