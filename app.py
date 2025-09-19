import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# -------------------------
# Helpers
# -------------------------
def safe_to_float(x):
    """Try several ways to coerce x to a Python float. Return np.nan on failure."""
    try:
        # if x is array-like (numpy scalar etc.)
        arr = np.array(x)
        # handle shape () or (1,) or (1,1) etc.
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        # if multiple values, take the first (we expect single-row predictions)
        return float(arr.reshape(-1)[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            try:
                return float(pd.to_numeric(x, errors="coerce"))
            except Exception:
                return np.nan

# -------------------------
# Page + style
# -------------------------
st.set_page_config(page_title="🌑 Dark Agri Predictor", page_icon="🌾", layout="centered")
st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(180deg, #060609 0%, #0b1020 100%); color: #E6EEF3; scroll-behavior: smooth; padding: 1rem 2rem; }
        h1 { color: #7BE4D4; text-align: center; margin-bottom: 0.1rem;}
        .subtitle { color: #9FB8C8; text-align: center; margin-top: 0rem; margin-bottom: 1rem;}
        label { color: #BFDCE8 !important; font-weight: 600; }
        .stTextInput input, .stNumberInput input, .stSelectbox > div { border-radius: 8px; background-color: #0f1720; color: #E6EEF3; }
        .stButton button { background: linear-gradient(90deg,#ff7a59,#c94bff); color: white; font-weight: 700; border-radius: 10px; padding: 8px 18px; }
        .result_val { color: #7BE4D4; font-weight: 700; font-size: 16px; }
        .small_note { color: #98b1c1; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Load model (pipeline)
# -------------------------
try:
    with open("Yield_prediction.pk1", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error("Error loading model file 'Yield_prediction.pk1'. Make sure it exists in the same folder.")
    st.exception(e)
    st.stop()

# -------------------------
# Header + Inputs
# -------------------------
st.title("🌑 Dark Agri Predictor")
st.markdown('<div class="subtitle">Compact dark UI — displays model outputs and recalculated totals</div>', unsafe_allow_html=True)

state_options = ["Andhra Pradesh","Bihar","Gujarat","Karnataka","Maharashtra","Punjab","Rajasthan","Tamil Nadu","Uttar Pradesh","West Bengal","Other"]
crop_options = ["Rice","Wheat","Maize","Cotton","Sugarcane","Soybean","Pulses","Other"]

col1, col2 = st.columns([1, 1])
with col1:
    crop_sel = st.selectbox("🌾 Crop", crop_options)
    crop = st.text_input("Type Crop (manual)") if crop_sel == "Other" else crop_sel

    season = st.selectbox("☀️ Season", ["Kharif","Rabi","Summer","Whole Year"])
    state_sel = st.selectbox("🏞️ State", state_options)
    state = st.text_input("Type State (manual)") if state_sel == "Other" else state_sel

with col2:
    area = st.number_input("📏 Area (hectares)", min_value=0.0, step=0.1, value=1.0, format="%.3f")
    annual_rain = st.number_input("🌧️ Annual_Rainfall (mm)", min_value=0.0, step=0.1, value=500.0)
    temperature = st.number_input("🌡️ temperature (°C)", value=25.0, step=0.1)
    humidity = st.number_input("💧 humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

col3, col4 = st.columns([1,1])
with col3:
    ph = st.number_input("🧪 ph", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    n_percent = st.number_input("🟢 N_percent (%)", min_value=0.0, step=0.1, value=0.5)
with col4:
    p_percent = st.number_input("🔴 P_percent (%)", min_value=0.0, step=0.1, value=0.2)
    k_percent = st.number_input("🟡 K_percent (%)", min_value=0.0, step=0.1, value=0.3)

soil_type = st.selectbox("🌍 Soil_Type", ["Alluvial","Black","Red","Laterite","Sandy","Clay"])
st.markdown('<div class="small_note">Choose from dropdown or type custom Crop/State above.</div>', unsafe_allow_html=True)

# -------------------------
# Predict button
# -------------------------
if st.button("🔎 Predict"):
    # prepare input dataframe
    input_df = pd.DataFrame([{
        "Crop": crop,
        "Season": season,
        "State": state,
        "Area": area,
        "Annual_Rainfall": annual_rain,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "N_percent": n_percent,
        "P_percent": p_percent,
        "K_percent": k_percent,
        "Soil_Type": soil_type
    }])

    try:
        raw_pred = model.predict(input_df)
    except Exception as e:
        st.error("Model prediction failed. Make sure your pipeline accepts a DataFrame with these column names.")
        st.exception(e)
        st.stop()

    # turn prediction into 1D numpy and coerce elements to float safely
    if isinstance(raw_pred, pd.DataFrame):
        pred_row = raw_pred.iloc[0].values
    else:
        pred_row = np.array(raw_pred).ravel()

    # require 6 outputs
    if pred_row.size != 6:
        st.error(f"Model returned {pred_row.size} outputs but 6 were expected. Raw output: {pred_row}")
        st.stop()

    # parse safely to floats
    yield_per_ha = safe_to_float(pred_row[0])   # 'Yield' (t/ha)
    total_yield = safe_to_float(pred_row[1])    # 'total_yield' (model t)
    fert_per_ha = safe_to_float(pred_row[2])    # 'Fertilizer_per_ha' (kg/ha)
    pest_per_ha= safe_to_float(pred_row[3])    # 'Pesticide_per_ha' (kg/ha or L/ha)
    fertilizer = safe_to_float(pred_row[4])    # 'Fertilizer' (model total kg)
    pesticide = safe_to_float(pred_row[5])     # 'Pesticide' (model total kg)

    # Ensure area is numeric
    try:
        area_f = float(area)
    except Exception:
        st.error("Area must be numeric.")
        st.stop()

    # -------------------------
    # Calculations (per-ha × area)
    # -------------------------
    total_yield = yield_per_ha * area_f
    fertilizer = fert_per_ha * area_f
    pesticide = pest_per_ha * area_f

    # -------------------------
    # Display results
    # -------------------------
    st.markdown("---")
 
    # Calculated outputs
    st.write(f"- Yield per ha        : `{yield_per_ha:.3f}` t/ha")
    st.write(f"- Total Yield   : `{total_yield:.3f}` t ")
    st.write(f"- Fertilizer per ha   : `{fert_per_ha:.3f}` kg/ha")
    st.write(f"- Total Fertilizer    : `{fertilizer:.3f}` kg ")
    st.write(f"- Pesticide per ha    : `{pest_per_ha:.3f}` kg/ha")
    st.write(f"- Total Pesticide     : `{pesticide:.3f}` kg ")

    # Comparison chart
    comp_df = pd.DataFrame({
        "Metric": ["Total Yield (t)", "Fertilizer (kg)", "Pesticide (kg)"],
        "Calculated": [total_yield, fertilizer, pesticide]
    })
    comp_melt = comp_df.melt(id_vars="Metric", var_name="Source", value_name="Value")

    fig = px.bar(
        comp_melt,
        x="Metric",
        y="Value",
        color="Source",
        barmode="group",
        text=comp_melt["Value"].round(2),
        template="plotly_dark",
        color_discrete_map={"Model": "#ff7a59", "Calculated": "#7BE4D4"},
    )
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), legend_title_text="")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ Done — calculations shown above.")
