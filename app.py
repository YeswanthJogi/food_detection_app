import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tempfile
import os

st.set_page_config(
    page_title="Food Detection and Calorie Estimation",
    page_icon="🍽️",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.title("🍽️ Food Detection and Calorie Estimation")
st.success("📷 Upload a food image using the sidebar to start food detection")

# -----------------------------
# Sidebar
# -----------------------------
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg","jpeg","png"])

# -----------------------------
# Calories Data
# -----------------------------
calorie_dict = {
    "apple":95,
    "banana":105,
    "pizza":285,
    "burger":354
}

# -----------------------------
# Load Model (FIXED)
# -----------------------------
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    
    MODEL_PATH = os.path.join(os.getcwd(), "best.pt")
    model = YOLO(MODEL_PATH)
    
    return model

model = load_model()

# -----------------------------
# Detection
# -----------------------------
if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        temp_path = temp.name

    results = model.predict(source=temp_path, conf=confidence)

    result = results[0]
    plotted = result.plot()

    st.image(plotted[:, :, ::-1], caption="Detection Output")

    names = model.names

    detections = []

    if result.boxes is not None:

        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for c, conf_score in zip(cls_ids, confs):
            detections.append({
                "Food Item": names[c],
                "Confidence": round(float(conf_score), 3)
            })

        df = pd.DataFrame(detections)

        st.subheader("Detected Items")
        st.dataframe(df)

        total_calories = 0
        for food in df["Food Item"]:
            total_calories += calorie_dict.get(food, 50)

        st.subheader(f"🔥 Total Calories: {total_calories} kcal")

    else:
        st.warning("No food detected")

    os.remove(temp_path)