# YOLOv8 Object Detection
## Setup
conda create -n yolo_env python=3.10 -y
conda activate yolo_env
pip install -r requirements.txt
## Run
streamlit run app.py
## code:

app.py
```
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
```
requirment.txt:
```
streamlit
ultralytics
opencv-python-headless
pillow
numpy
```
README.md:
```
# YOLOv8 Object Detection
## Setup
conda create -n yolo_env python=3.10 -y
conda activate yolo_env
pip install -r requirements.txt
## Run
streamlit run app.py
```
## output:

<img width="1694" height="827" alt="Screenshot 2026-09-05 204141" src="https://github.com/user-attachments/assets/5544ba53-3953-4996-be50-25e3fcb09fe2" />


<img width="1907" height="956" alt="Screenshot 2026-09-05 204157" src="https://github.com/user-attachments/assets/05dff749-9667-49c5-aead-10ff052d9391" />

## Result:
