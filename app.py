import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
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
# Styling
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
}
.subtitle {
text-align:center;
font-size:18px;
color:#8aa0b4;
}
.banner {
background:linear-gradient(135deg,#1f4037,#99f2c8);
padding:12px;
border-radius:12px;
text-align:center;
font-weight:600;
color:black;
margin-top:10px;
}
.card{
background:linear-gradient(135deg,#667eea,#764ba2);
padding:10px;
border-radius:10px;
text-align:center;
color:white;
margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title">🍽️ FOOD DETECTION</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload image and detect food items</div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25)

uploaded_file = st.sidebar.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

# -----------------------------
# Banner
# -----------------------------
if uploaded_file is None:
    st.markdown('<div class="banner">Upload image to start detection</div>', unsafe_allow_html=True)

# -----------------------------
# Calories
# -----------------------------
calorie_dict = {
    "apple":95,
    "banana":105,
    "orange":62,
    "pizza":285,
    "burger":354
}

# -----------------------------
# Load Model (SAFE)
# -----------------------------
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")   # default safe model

# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # 🔥 FIXED PART (NO TEMP FILE)
    img_array = np.array(image)

    model = load_model()

    results = model.predict(source=img_array, conf=confidence)

    result = results[0]
    plotted = result.plot()

    with col2:
        st.image(plotted[:, :, ::-1], caption="Detection Output", use_container_width=True)

    # -----------------------------
    # Results
    # -----------------------------
    names = model.names

    if result.boxes is not None and len(result.boxes) > 0:

        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        detections = []

        for c, conf_score in zip(cls_ids, confs):
            detections.append({
                "Food Item": names[c],
                "Confidence": round(float(conf_score), 3)
            })

        df = pd.DataFrame(detections)

        st.subheader("Detected Items")

        counts = df["Food Item"].value_counts()

        for food, count in counts.items():
            st.markdown(f"""
            <div class="card">
            <h3>{food}</h3>
            <p>{count} items</p>
            </div>
            """, unsafe_allow_html=True)

        # -----------------------------
        # Calories
        # -----------------------------
        nutrition = []

        for food, count in counts.items():
            cal = calorie_dict.get(food, 50)
            nutrition.append({
                "Food": food,
                "Count": count,
                "Calories": cal * count
            })

        nutrition_df = pd.DataFrame(nutrition)

        st.dataframe(nutrition_df)

        total = nutrition_df["Calories"].sum()

        st.success(f"🔥 Total Calories: {total} kcal")

        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie(nutrition_df["Calories"], labels=nutrition_df["Food"], autopct="%1.1f%%")
        st.pyplot(fig)

    else:
        st.warning("No objects detected")