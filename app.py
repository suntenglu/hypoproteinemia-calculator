import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Persistent Hypoproteinemia Risk Calculator", layout="centered")

MODEL_PATH = "best_model__LGBM_LightGBM.joblib"
model = joblib.load(MODEL_PATH)

# 模型真实输入列名（你的模型是 Column_0 ~ Column_6）
MODEL_FEATURES = [f"Column_{i}" for i in range(7)]

# 展示给用户看的变量（顺序必须与 Column_0..6 对应）
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
st.caption("Model: LightGBM (LGBMClassifier). Outcome: Persistent Hypoproteinemia")

with st.sidebar:
    st.header("Risk Stratification")
    low_cut = st.number_input("Low-risk cutoff (<=)", value=0.30, min_value=0.0, max_value=1.0, step=0.01)
    high_cut = st.number_input("High-risk cutoff (>=)", value=0.60, min_value=0.0, max_value=1.0, step=0.01)
    st.write("Rule: p ≤ low_cut → Low; p ≥ high_cut → High; otherwise → Intermediate")
    st.divider()
    st.caption("Note: Inputs must use the same units as model training data.")
    st.caption("This tool is for research/educational use and does not replace clinical judgment.")

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
    pct = p * 100

    if p <= low_cut:
        tier = "Low risk"
        badge = "🟢"
    elif p >= high_cut:
        tier = "High risk"
        badge = "🔴"
    else:
        tier = "Intermediate risk"
        badge = "🟠"

    st.subheader("Result")
    st.metric("Predicted risk", f"{pct:.1f}%")
    st.progress(min(max(p, 0.0), 1.0))
    st.write(f"Risk category: **{badge} {tier}**")

    st.subheader("Input summary")
    # 给用户看英文变量名（更专业）
    show_df = pd.DataFrame({
        "Predictor": [x[0] for x in FEATURE_MAP],
        "Value": [vals[f"Column_{i}"] for i in range(7)],
        "Unit": [x[1] for x in FEATURE_MAP],
    })
    st.dataframe(show_df, use_container_width=True)

    # 导出本次结果
    export_df = show_df.copy()
    export_df["Predicted_risk_probability"] = p
    export_df["Risk_category"] = tier
    csv = export_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download result (CSV)", csv, file_name="risk_result.csv", mime="text/csv")