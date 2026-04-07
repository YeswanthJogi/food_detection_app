import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Food Detection and Calorie Estimation",
    page_icon="🍽️",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
.title {
font-size:42px;
font-weight:800;
text-align:center;
background: linear-gradient(90deg,#ff416c,#ff4b2b,#ffb347);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-bottom:5px;
}
.subtitle {
text-align:center;
font-size:18px;
color:#8aa0b4;
margin-bottom:5px;
}
.banner {
background:linear-gradient(135deg,#1f4037,#99f2c8);
padding:12px;
border-radius:12px;
text-align:center;
font-size:16px;
font-weight:600;
color:black;
margin-top:10px;
margin-bottom:20px;
}
.card{
background:linear-gradient(135deg,#667eea,#764ba2);
padding:10px;
border-radius:10px;
text-align:center;
color:white;
margin-bottom:8px;
}
.sidebar-title{
font-size:22px;
font-weight:700;
text-align:center;
margin-bottom:10px;
color:#ff7a18;
}
.sidebar-box{
background:linear-gradient(135deg,#1c1c1c,#2c3e50);
padding:10px;
border-radius:10px;
margin-bottom:10px;
color:white;
text-align:center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title">🍽️ FOOD DETECTION AND CALORIE ESTIMATION</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a food image and detect items using your trained YOLO model</div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown('<div class="sidebar-title">⚙ Detection Settings</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-box">Adjust confidence level for detection</div>', unsafe_allow_html=True)

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

st.sidebar.markdown("### 📷 Upload Food Image")
uploaded_file = st.sidebar.file_uploader("Browse Image", type=["jpg","jpeg","png"], label_visibility="collapsed")
st.sidebar.caption("Supported formats: JPG • JPEG • PNG")

# -----------------------------
# Banner
# -----------------------------
if uploaded_file is None:
    st.markdown('<div class="banner">📷 Upload a food image using the sidebar to start food detection</div>', unsafe_allow_html=True)

# -----------------------------
# Calories
# -----------------------------
calorie_dict = {
    "apple":95,
    "banana":105,
    "grape":3,
    "orange":62,
    "pizza":285,
    "burger":354
}

# -----------------------------
# Load Model (SAFE VERSION)
# -----------------------------
import torch

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    return model

# -----------------------------
# Main Logic
# -----------------------------
if uploaded_file is not None:

    st.subheader("🔍 Food Detection Results")

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        temp_path = temp.name

    # Load model only when needed
    model = load_model()

    # Prediction
    results = model(img_array)

    results.render()   # draw boxes
    output_image = results.ims[0]

    with col2:
        st.subheader("🎯 Detection Output")
        st.image(output_image, caption="Detection Output", use_container_width=True)
    names = model.names
    detections = []

    if result.boxes is not None and len(result.boxes) > 0:

        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for c, conf_score in zip(cls_ids, confs):
            detections.append({
                "Food Item": names[c],
                "Confidence": round(float(conf_score), 3)
            })

        df = pd.DataFrame(detections)

        # -----------------------------
        # Cards
        # -----------------------------
        st.subheader("🍎 Detected Food Items")

        counts = df["Food Item"].value_counts()

        for food, count in counts.items():
            avg_conf = df[df["Food Item"] == food]["Confidence"].mean() * 100

            st.markdown(f"""
            <div class="card">
            <h4>{food.capitalize()}</h4>
            <h3>{count} item(s)</h3>
            <p>{avg_conf:.1f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)

        # -----------------------------
        # Nutrition Table
        # -----------------------------
        st.subheader("📊 Nutrition Table")

        nutrition = []

        for food, count in counts.items():
            calories = calorie_dict.get(food, 50)

            nutrition.append({
                "Food Item": food,
                "Count": count,
                "Calories": calories * count
            })

        nutrition_df = pd.DataFrame(nutrition)
        st.dataframe(nutrition_df, use_container_width=True)

        # -----------------------------
        # Total Calories
        # -----------------------------
        total_calories = nutrition_df["Calories"].sum()

        st.markdown(
        f"<h2 style='color:#ff4b2b'>🔥 Total Estimated Calories: {total_calories} kcal</h2>",
        unsafe_allow_html=True
        )

        # -----------------------------
        # Pie Chart
        # -----------------------------
        st.subheader("🥧 Calorie Distribution")

        fig, ax = plt.subplots()
        ax.pie(
            nutrition_df["Calories"],
            labels=nutrition_df["Food Item"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)

    else:
        st.warning("No food items detected.")

    os.remove(temp_path)