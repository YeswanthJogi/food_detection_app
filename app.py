import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

st.set_page_config(
    page_title="Food Detection and Calorie Estimation",
    page_icon="🍽️",
    layout="wide"
)

# -----------------------------
# UI Styling (same as yours)
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

st.markdown(
'<div class="subtitle">Upload a food image and detect items using your trained YOLO model</div>',
unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown('<div class="sidebar-title">⚙ Detection Settings</div>', unsafe_allow_html=True)

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.markdown(
    '<div class="banner">📷 Upload a food image using the sidebar to start detection</div>',
    unsafe_allow_html=True
    )

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
@st.cache_resource
def load_model():
    import sys
    sys.modules['cv2'] = None   # 🔥 BLOCK cv2 (IMPORTANT)

    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# Detection
# -----------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image")

    results = model.predict(image, conf=confidence)

    plotted = results[0].plot()

    with col2:
        st.image(plotted[:, :, ::-1], caption="Detection Output")

    names = model.names
    detections = []

    if results[0].boxes is not None:

        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for c, conf_score in zip(cls_ids, confs):
            detections.append({
                "Food Item": names[c],
                "Confidence": round(float(conf_score), 3)
            })

        df = pd.DataFrame(detections)

        st.subheader("Detected Items")
        st.dataframe(df)

        counts = df["Food Item"].value_counts()

        nutrition = []
        for food, count in counts.items():
            cal = calorie_dict.get(food, 50)
            nutrition.append({
                "Food Item": food,
                "Count": count,
                "Calories": cal * count
            })

        nutrition_df = pd.DataFrame(nutrition)

        st.subheader("Nutrition")
        st.dataframe(nutrition_df)

        total = nutrition_df["Calories"].sum()
        st.success(f"🔥 Total Calories: {total} kcal")

        fig, ax = plt.subplots()
        ax.pie(nutrition_df["Calories"], labels=nutrition_df["Food Item"], autopct="%1.1f%%")
        st.pyplot(fig)

    else:
        st.warning("No objects detected")