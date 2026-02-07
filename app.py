import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Persistent Hypoproteinemia Risk Calculator", layout="centered")

MODEL_PATH = "best_model__LGBM_LightGBM.joblib"
model = joblib.load(MODEL_PATH)

# 模型真实列名
MODEL_FEATURES = [f"Column_{i}" for i in range(7)]

# 显示变量
FEATURE_MAP = [
    ("Prognostic Nutritional Index (PNI)", "", 45.0),
    ("Lymphocyte Percentage", "%", 25.0),
    ("Prealbumin (PA)", "g/L", 200.0),
    ("Neutrophil-to-Lymphocyte Ratio (NLR)", "", 2.5),
    ("Carcinoembryonic Antigen (CEA)", "ng/mL", 3.0),
    ("Alpha-fetoprotein (AFP)", "ng/mL", 10.0),
    ("High-sensitivity C-reactive Protein (hs-CRP)", "mg/L", 5.0),
]

st.title("Persistent Hypoproteinemia Risk Calculator")
st.caption("Model: LightGBM (LGBMClassifier)")

st.subheader("Enter predictors")

vals = {}
col1, col2 = st.columns(2)

for i, (name, unit, default) in enumerate(FEATURE_MAP):
    shown = f"{name} ({unit})" if unit else name
    if i % 2 == 0:
        vals[f"Column_{i}"] = col1.number_input(shown, value=float(default))
    else:
        vals[f"Column_{i}"] = col2.number_input(shown, value=float(default))

st.divider()

if st.button("Calculate risk", type="primary"):
    X = pd.DataFrame([[vals[c] for c in MODEL_FEATURES]], columns=MODEL_FEATURES)
    p = float(model.predict_proba(X)[0, 1])

    st.success(f"Predicted probability of persistent hypoproteinemia: **{p:.3f}**")

    st.subheader("Input summary")
    st.dataframe(X, use_container_width=True)
