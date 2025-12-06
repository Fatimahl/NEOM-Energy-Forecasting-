import streamlit as st
import base64
import joblib
import pandas as pd
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# =========================
# إعداد الصفحة + العنوان
# =========================
st.set_page_config(page_title="⚡ NEOM Energy Forecast ⚡", layout="centered")

# 🔮 تنسيق الأزرار (Next + Predict) بنفس درجة البنفسجي في الخلفية
st.markdown(
    """
    <style>
    div.stButton > button {
        width: 70%;
        margin: 20px auto 10px auto;
        display: block;
        padding: 16px 0;
        font-size: 20px;
        font-weight: 600;
        background: linear-gradient(90deg, #4C1D57, #7B2CBF);
        color: #FDFDFD;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        cursor: pointer;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #7B2CBF, #4C1D57);
        box-shadow: 0px 0px 18px rgba(123, 44, 191, 0.6);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# إدارة الصفحات
# =========================
if "page" not in st.session_state:
    st.session_state.page = "intro"

def go_to_main():
    st.session_state.page = "main"
    st.rerun()

# =========================
# الخلفية
# =========================
with open("gg.jpg", "rb") as img_file:
    b64_string = base64.b64encode(img_file.read()).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{b64_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    .stSelectbox, .stRadio, .stNumberInput {{
        background-color: rgba(0,0,0,0.6);
        border-radius: 12px;
        padding: 6px;
    }}

    h1, h2, h3, label, div {{
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# تحميل ملفات المودل
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    base_row = joblib.load("base_row.pkl")
    y_mean, y_std = joblib.load("target_stats.pkl")
    return model, scaler, feature_cols, base_row, y_mean, y_std

model, scaler, feature_cols, base_row, y_mean, y_std = load_artifacts()

season_map = {
    "Winter ❄": 0,
    "Spring 🌸": 1,
    "Summer ☀": 2,
    "Autumn 🍁": 3,
}

# =========================
# دالة: تطبيق نفس الـ StandardScaler
# =========================
def set_scaled_value(row, col_name, raw_value):
    """نفس السكيلر اللي استخدم وقت التدريب."""
    if raw_value is None:
        return
    try:
        feature_names = list(scaler.feature_names_in_)
    except AttributeError:
        feature_names = []
    if col_name in feature_names:
        idx = feature_names.index(col_name)
        mean = scaler.mean_[idx]
        scale = scaler.scale_[idx] if scaler.scale_[idx] != 0 else 1.0
        row[col_name] = (raw_value - mean) / scale
    else:
        row[col_name] = raw_value

# =========================
# صفحة المقدّمة
# =========================
def render_intro():
    st.markdown(
        "<h1 style='text-align:center;'>NEOM Energy – Renewable Load Forecasting ⚡</h1>",
        unsafe_allow_html=True
    )

    # وصف بسيط للمشروع
    st.markdown(
        """
        ### NEOM Green Energy: An AI-based system designed to forecast electricity consumption in NEOM areas, enabling optimized distribution of renewable energy and minimizing energy loss. 
        """
    )

    # اللوقو
    try:
        logo = Image.open("nnn.PNG")
        st.image(logo, width=180)
    except:
        pass

    # ====== Intro Video  ======
    try:
        st.video("ooo.mp4")
    except:
        st.warning("Intro video not found.")

    st.markdown(
        "Use the inputs in the next page to simulate different scenarios and see how the load changes."
    )

    st.markdown("---")
    if st.button("Next ➜ | التالي"):
        go_to_main()

# =========================
# صفحة المدخلات + التنبؤ
# =========================
def render_main():
    st.markdown(
        "<h2 style='text-align:center;'>Scenario Inputs & Smart Forecast ⚡</h2>",
        unsafe_allow_html=True
    )

    # واجهة الإدخال (مبنية على أهم الـ Features)
    is_holiday = st.radio("Is Holiday?", ["Yes", "No"])
    season_label = st.selectbox("Season", ["Winter ❄", "Spring 🌸", "Summer ☀", "Autumn 🍁"])

    # حالة القطاع والمنطقة
    load_sector = st.selectbox(
        "Load Sector Type | نوع القطاع",
        ["Commercial", "Industrial", "Residential"]
    )

    area_type = st.selectbox(
        "Area Type | نوع المنطقة",
        ["Urban", "Suburban", "Rural"]
    )

    # حالة الشبكة
    current_level = st.number_input(
        "Current Level (A) | شدة التيار الحالية (A)",
        value=100.0
    )

    peak_load = st.radio(
        "Current Load Level (Normal / Peak) | مستوى استهلاك الكهرباء الآن (عادي / ذروة)",
        ["Normal", "Peak"]
    )

    curtail_flag = st.radio(
        "Energy Status (Waste or Shortage) | حالة الطاقة (هدر أو نقص)",
        ["No Event", "Event"]
    )

    # مصادر الطاقة المتجددة
    solar_pv = st.number_input(
        "Solar Power Generation (kW) | إنتاج الطاقة الشمسية (kW)",
        value=50.0
    )

    wind_power = st.number_input(
        "Wind Power Generation (kW) | إنتاج طاقة الرياح (kW)",
        value=20.0
    )

    # الطقس
    load_weather = st.selectbox(
        "Weather Condition | حالة الطقس",
        ["clear", "cloudy", "rainy", "snowy", "stormy"]
    )

    st.markdown("---")

    # زر التنبؤ
    if st.button("Predict Load ⚡"):

        # نبدأ من base_row
        row = base_row.copy()

        # 🕒 الوقت تلقائي لتقليل الازرار على المستخدم
        now = datetime.now()
        set_scaled_value(row, "Hour of Day", now.hour)
        set_scaled_value(row, "Day of Week", now.weekday())
        set_scaled_value(row, "Month", now.month)
        set_scaled_value(row, "Is Weekend", 1 if now.weekday() in [5, 6] else 0)

        # الموسم + الإجازة
        set_scaled_value(row, "Season", season_map[season_label])
        set_scaled_value(row, "Is Holiday", 1 if is_holiday == "Yes" else 0)

        # 🟦 القيم من المستخدم
        set_scaled_value(row, "Current Level (A)", current_level)
        set_scaled_value(row, "Solar PV Output (kW)", solar_pv)
        set_scaled_value(row, "Wind Power Output (kW)", wind_power)

        # Peak / Curtailment → ثنائية
        row["Peak Load Indicator"] = 1 if peak_load == "Peak" else 0
        row["Curtailment Risk / Surplus Flag"] = 1 if curtail_flag == "Event" else 0

        # One-Hot للقطاع
        for c in ["Commercial", "Industrial", "Residential"]:
            row[c] = 0
        row[load_sector] = 1

        # One-Hot للمنطقة
        for c in ["Urban", "Suburban", "Rural"]:
            row[c] = 0
        row[area_type] = 1

        # One-Hot للطقس
        weather_cols = ["Clear", "Cloudy", "Rainy", "Snowy", "Stormy"]
        for c in weather_cols:
            row[c] = 0

        weather_map = {
            "clear": "Clear",
            "cloudy": "Cloudy",
            "rainy": "Rainy",
            "snowy": "Snowy",
            "stormy": "Stormy",
        }
        row[weather_map[load_weather.lower()]] = 1

        # DataFrame بنفس ترتيب أعمدة المودل
        row_df = pd.DataFrame([row])
        model_features = model.get_booster().feature_names
        row_df = row_df.reindex(columns=model_features, fill_value=0)

        # التنبؤ (مقَيّس → kW)
        prediction_scaled = model.predict(row_df)[0]
        prediction_real = prediction_scaled * y_std + y_mean
        prediction_real = max(prediction_real, 0)

        # ========= 1) كرت النتيجة الأساسي + التصنيف =========
        base_load = prediction_real

        if base_load < 650:
            user_color = "#2ecc71"   # أخضر
            user_label = "Low Load"
            user_emoji = "🟢"
        elif base_load < 800:
            user_color = "#3498db"   # أزرق
            user_label = "Medium Load"
            user_emoji = "🟡"
        else:
            user_color = "#e74c3c"   # أحمر
            user_label = "High Load"
            user_emoji = "🔴"

        badge_html = f"""
        <div style="
            padding: 16px;
            border-radius: 12px;
            background-color: {user_color};
            color: white;
            margin-top: 10px;
            margin-bottom: 10px;">
            <h3 style="margin:0;">{user_emoji} Load Status: {user_label}</h3>
            <p style="margin:4px 0 0 0;">Predicted Electricity Load: <b>{base_load:,.2f} kW</b></p>
        </div>
        """
        st.markdown(badge_html, unsafe_allow_html=True)

        st.success(f"Predicted Electricity Load: {prediction_real:,.2f} kW ⚡")

        # ========= 2)  تحليلات ذكية =========
        insights = []

        if peak_load == "Peak":
            insights.append("• The system is currently in a peak demand period (high usage).")
        else:
            insights.append("• The system is currently in a normal demand period.")

        if curtail_flag == "Event":
            insights.append("• There is an energy waste/shortage event, which affects grid stability.")
        else:
            insights.append("• No major energy waste/shortage event detected.")

        if solar_pv > 80:
            insights.append("• High solar generation is helping reduce net load. ☀")
        elif solar_pv < 20:
            insights.append("• Low solar generation, the grid depends more on other sources.")

        if wind_power > 50:
            insights.append("• Strong wind power contribution detected. 🌬")
        elif wind_power < 10:
            insights.append("• Wind contribution is very low.")

        if load_weather in ["stormy", "rainy"]:
            insights.append("• Weather conditions (rainy/stormy) may increase uncertainty in demand.")
        elif "sunny" in load_weather or load_weather in ["clear"]:
            insights.append("• Clear weather conditions are generally more predictable for the grid.")

        st.markdown("### 🔎 Intelligent Insights | تحليلات ذكية")
        for line in insights:
            st.markdown(line)

        # ========= 3) Load Comparison Plot (Low / Your / High) =========
        low_load = base_load * 0.75
        high_load = base_load * 1.10

        labels = ["Low Demand", "Your Scenario", "High Demand"]
        values = [low_load, base_load, high_load]

        colors = [
            "#2ecc71",   # Low → أخضر
            user_color,  # Your Scenario → حسب التصنيف
            "#e74c3c"    # High → أحمر
        ]

        st.markdown("### 📊 Load Comparison | مقارنة الاحتمالات")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Approx. Electricity Load (kW)")

        for i, v in enumerate(values):
            ax.text(i, v + max(values)*0.02, f"{v:,.0f}", ha="center")

        st.pyplot(fig)

        # ========= 4) Energy Efficiency Advice | نصائح تحسين الاستهلاك =========
        st.markdown("### 💡 Energy Efficiency Advice | نصائح لتحسين استهلاك الطاقة")

        tips = []
        if user_label == "High Load":
            tips.append("• Try shifting heavy loads (e.g., EV charging, industrial machines) to off-peak hours.")
            tips.append("• Consider increasing solar / wind utilization to reduce grid stress.")
            tips.append("• Use smart control to turn off non-critical loads.")
        elif user_label == "Medium Load":
            tips.append("• Your load is moderate. Optimizing AC usage and lighting can still save energy.")
            tips.append("• Maintain current renewable contribution and monitor peak events.")
        else:
            tips.append("• Your load level is efficient and healthy for the grid ✅.")
            tips.append("• Keep using renewable sources and avoid unnecessary spikes in demand.")

        for t in tips:
            st.markdown(t)

# =========================
# تشغيل الصفحة المناسبة
# =========================
if st.session_state.page == "intro":
    render_intro()
else:
    render_main()