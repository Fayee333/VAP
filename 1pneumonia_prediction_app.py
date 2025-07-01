# pneumonia_prediction_app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from loguru import logger

# Page configuration
st.set_page_config(
    page_title="Postoperative Pneumonia Prediction APP",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Feature name mapping
FEATURE_MAPPING = {
    'EF': 'Ejection Fraction (%)',
    'CPB': 'Cardiopulmonary Bypass Time (min)',
    'SCr': 'Serum Creatinine (μmol/L)',
    'BL': 'Intraoperative Blood Loss (mL)',
    'Gender': 'Gender',
    'PWR': 'Platelet/WBC Ratio',
    'TBIL': 'Total Bilirubin (μmol/L)'
}

# Model loading
@st.cache_resource
def load_model():
    try:
        model = joblib.load(r'C:\Users\fcy\my_model.pkl')  # Update with your model path
        logger.success("Model loaded successfully")
        return model
    except FileNotFoundError:
        logger.error("Model file not found")
        st.error("❌ Model file not found - please check path configuration")
        st.stop()
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

# User input interface
def user_input_features():
    with st.sidebar:
        st.header("⚕️ Patient Parameters Input")
        
        with st.expander("Hemodynamic Indicators", expanded=True):
            ef = st.slider('Ejection Fraction (%)', 30, 100, 60,
                         help="Cardiac function indicator (normal range: 50-70%)")
            cpb = st.number_input('CPB Time (minutes)', 0, 600, 120, step=5,
                                format="%d", help="Duration of cardiopulmonary bypass")
            scr = st.number_input('Serum Creatinine (μmol/L)', 20.0, 500.0, 80.0, step=5.0,
                                format="%.1f", help="Renal function marker (normal: M 53-106, F 44-97)")
        
        with st.expander("Other Parameters"):
            bl = st.number_input('Blood Loss (mL)', 0, 5000, 500, step=50,
                               format="%d", help="Total intraoperative blood loss")
            gender = st.radio("Gender", ['Male', 'Female'], horizontal=True,
                            help="Biological sex")
            pwr = st.number_input('Platelet/WBC Ratio', 0.0, 50.0, 20.0, step=0.5,
                                format="%.1f", help="Inflammatory marker (normal range: 10-30)")
            tbil = st.number_input('Total Bilirubin (μmol/L)', 5.0, 300.0, 20.0, step=5.0,
                                 format="%.1f", help="Liver function marker (normal: 3.4-20.5)")

    return pd.DataFrame([[ef, cpb, scr, bl, 1 if gender == 'Male' else 0, pwr, tbil]],
                      columns=FEATURE_MAPPING.keys())

# 修改后的 SHAP 可视化函数
def plot_shap_explanation(model, input_df):
    try:
        # 初始化解释器
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(input_df)
        
        # 二分类/多分类适配
        if isinstance(shap_values, list):
            # 多分类：选择肺炎类别（假设类别1为肺炎）
            base_value = explainer.expected_value[1]
            shap_vals = shap_values[1]
        else:
            # 二分类：直接使用
            base_value = explainer.expected_value
            shap_vals = shap_values
        
        # 创建可视化
        plt.figure(figsize=(12, 6), facecolor='white')  # 强制白底
        shap.force_plot(
            base_value=base_value,
            shap_values=shap_vals,
            features=input_df,
            feature_names=[FEATURE_MAPPING[c] for c in input_df.columns],
            matplotlib=True,
            show=False,
            text_rotation=15,
            plot_cmap=['#ff0051', '#008bfb']  # 自定义颜色
        )
        
        # 优化显示
        plt.tight_layout()
        plt.gcf().set_facecolor('white')
        plt.axis('off')  # 隐藏坐标轴
        
        return plt.gcf()
    
    except Exception as e:
        st.error(f"特征解释生成失败: {str(e)}")
        return None

# Main interface
def main():
    st.title("Postoperative pneumonia after cardiac surgery")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    # Get input
    input_df = user_input_features()
    
    # Display parameters
    with st.expander("📋 Current Input Parameters", expanded=True):
        display_df = input_df.rename(columns=FEATURE_MAPPING)
        st.dataframe(display_df.T.style.format("{:.1f}"), use_container_width=True)
    
    # Prediction section
    if st.button("🚀 Start Risk Assessment", use_container_width=True, type="primary"):
        with st.spinner('Analyzing parameters...'):
            try:
                # Make prediction
                proba = model.predict_proba(input_df)[0][1]
                
                # Display results
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("📊 Risk Prediction")
                    risk_level = "High Risk" if proba > 0.5 else "Low Risk"
                    color = "#FF4B4B" if proba > 0.5 else "#2ECC71"
                    
                    st.markdown(f"""
                    <div style="border-radius: 10px; padding: 2rem; background: {color}10; 
                                border-left: 5px solid {color}; margin: 1rem 0;">
                        <h3 style="color: {color}; margin:0 0 1rem 0;">{risk_level}</h3>
                        <div style="font-size: 2.5rem; font-weight: bold; color: {color};">
                            {proba*100:.1f}%
                        </div>
                        <div style="margin-top: 1rem; color: #666;">
                            Decision Threshold: 50% (clinical judgment recommended)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("📈 Feature Contribution Analysis")
                    fig = plot_shap_explanation(model, input_df)
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        st.markdown("""
                        ​**Interpretation Guide**:
                        - → Red arrows indicate risk-increasing factors
                        - ← Blue arrows indicate risk-decreasing factors
                        - Arrow length represents impact magnitude
                        - Baseline Value: {:.2f} (model average prediction)
                        """.format(model.predict_proba(input_df)[0][1]))
                
                # Clinical recommendations
                st.markdown("---")
                st.subheader("🩺 Clinical Recommendations")
                if proba > 0.7:
                    st.warning("""
                    ​**High Risk Protocol**:
                    1. Enhanced respiratory monitoring
                    2. Consider prophylactic antibiotic therapy
                    3. Chest X-ray within 24 hours post-op
                    4. Continuous vital signs monitoring
                    """)
                elif proba > 0.5:
                    st.warning("""
                    ​**Moderate Risk Protocol**:
                    1. Incentive spirometry every 2 hours
                    2. Daily blood gas analysis
                    3. Strict fluid balance management
                    4. Pulmonary auscultation Q4H
                    """)
                else:
                    st.success("""
                    ​**Low Risk Protocol**:
                    1. Standard postoperative care
                    2. Early ambulation protocol
                    3. Maintain SpO₂ > 95%
                    4. Monitor for respiratory symptoms
                    """)

            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                st.error("Prediction error - please verify input parameters")

if __name__ == '__main__':
    main()