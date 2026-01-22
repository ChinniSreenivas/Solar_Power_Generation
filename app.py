import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ---------------- Page Config ----------------
st.set_page_config(page_title="Solar Power AI Dashboard", page_icon="☀️", layout="wide")

# ---------------- CSS Styling ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #000428, #004e92);
}

h1, h2, h3, h4, label, p {
    color: white !important;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ff416c, #ff4b2b);
}

.card {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    padding: 25px;
    border-radius: 18px;
    color: white;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.4);
    text-align: center;
}

.input-box {
    background: rgba(255,255,255,0.12);
    padding: 20px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown("""
<h1 style='text-align:center;'>☀️ Solar Power Generation AI Dashboard</h1>
<p style='text-align:center;'>Smart Prediction System using Machine Learning</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- Load & Train Models ----------------
@st.cache_resource
def train_models():
    df = pd.read_csv("solarpowergeneration.csv")

    target = [c for c in df.columns if "power" in c.lower()][0]
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    trained = {}
    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        trained[name] = pipe

    return trained, list(X.columns)

models, feature_cols = train_models()

# ---------------- Sidebar ----------------
st.sidebar.markdown("## ⚙️ Control Panel")

model_choice = st.sidebar.selectbox("🤖 Select ML Model", list(models.keys()))

distance_to_solar_noon = st.sidebar.slider("🌞 Distance to Solar Noon", 0.0, 2.0, 0.5)
temperature = st.sidebar.slider("🌡 Temperature", -10.0, 50.0, 25.0)
sky_cover = st.sidebar.selectbox("☁ Sky Cover", [0,1,2,3,4])
wind_speed = st.sidebar.slider("💨 Wind Speed", 0.0, 20.0, 3.0)
wind_direction = st.sidebar.slider("🧭 Wind Direction", 0.0, 360.0, 180.0)
visibility = st.sidebar.slider("👁 Visibility", 0.0, 20.0, 10.0)
humidity = st.sidebar.slider("💧 Humidity", 0.0, 100.0, 60.0)
average_wind_speed = st.sidebar.slider("📈 Avg Wind Speed", 0.0, 20.0, 3.0)
average_pressure = st.sidebar.slider("⏲ Avg Pressure", 25.0, 35.0, 30.0)

predict_btn = st.sidebar.button("⚡ Predict Now")

# ---------------- Main Layout ----------------
col1, col2, col3 = st.columns([2,1,1])

# Input Summary
with col1:
    st.markdown("### 📋 Weather Conditions")
    st.markdown("<div class='input-box'>", unsafe_allow_html=True)
    st.write({
        "Solar Noon Distance": distance_to_solar_noon,
        "Temperature": temperature,
        "Sky Cover": sky_cover,
        "Wind Speed": wind_speed,
        "Wind Direction": wind_direction,
        "Visibility": visibility,
        "Humidity": humidity,
        "Avg Wind Speed": average_wind_speed,
        "Avg Pressure": average_pressure
    })
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction Card
with col2:
    st.markdown("### 🔮 Prediction")

    if predict_btn:
        input_df = pd.DataFrame({
            "distance-to-solar-noon": [distance_to_solar_noon],
            "temperature": [temperature],
            "wind-direction": [wind_direction],
            "wind-speed": [wind_speed],
            "sky-cover": [sky_cover],
            "visibility": [visibility],
            "humidity": [humidity],
            "average-wind-speed-(period)": [average_wind_speed],
            "average-pressure-(period)": [average_pressure]
        })

        pred = models[model_choice].predict(input_df)[0]

        st.markdown(f"""
        <div class="card">
        ⚡ Power Output<br>
        <h2>{pred:.2f}</h2>
        Joules
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Click Predict")

# Prediction History Chart
with col3:
    st.markdown("### 📈 Prediction Trend")

    if "history" not in st.session_state:
        st.session_state.history = []

    if predict_btn:
        st.session_state.history.append(pred)

    if st.session_state.history:
        chart_df = pd.DataFrame({
            "Prediction": st.session_state.history
        })
        st.line_chart(chart_df)
    else:
        st.write("No predictions yet")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Solar Power Prediction System | ML + Streamlit Project | By Bhavana</p>",
    unsafe_allow_html=True
)
