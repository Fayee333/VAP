# pneumonia_prediction_app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="VAP Prediction APP",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 特征名称映射
FEATURE_MAPPING = {
     'EF': 'mean arterial pressure (mmHg)',
    'CPB': 'temperature',
    'SCr': 'Serum Creatinine (μmol/L)',
    'BL': 'age (year)',
    'Gender': 'Gender',
    'PWR': 'Platelet/WBC Ratio',
    'TBIL': 'Total Bilirubin (μmol/L)'
}

# ----------- 模型加载函数 -----------
# 优化了模型加载，使用相对路径，更易于部署
@st.cache_resource
def load_model():
    """健壮的模型加载函数，适用于不同部署环境"""
    try:
        # 尝试多种可能的模型位置（GitHub和Streamlit友好）
        possible_paths = [
            Path("models") / "my_model.pkl",        # GitHub推荐位置
            Path("my_model.pkl"),                   # 根目录位置
            Path("app") / "models" / "my_model.pkl",# 多层级项目
            Path("pneumonia_app") / "my_model.pkl", # Streamlit Cloud结构
            Path("resources") / "my_model.pkl"      # 资源文件夹
        ]
        
        # 尝试查找并加载模型
        for model_path in possible_paths:
            if model_path.exists():
                logger.info(f"找到模型文件: {model_path}")
                model = joblib.load(model_path)
                logger.info("模型加载成功")
                return model
        
        # 所有路径都找不到文件
        logger.error(f"未找到模型文件。检查位置: {[str(p) for p in possible_paths]}")
        st.error("❌ 模型文件未找到 - 请确认部署设置")
        
        # 添加模型上传作为后备方案
        st.subheader("模型上传备选方案")
        uploaded_file = st.file_uploader("上传my_model.pkl文件", type=["pkl", "joblib"])
        if uploaded_file:
            try:
                with st.spinner("处理上传文件中..."):
                    # 保存上传的模型
                    save_path = Path("uploaded_model.pkl")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    model = joblib.load(save_path)
                    st.success("模型上传并加载成功！")
                    return model
            except Exception as e:
                st.error(f"上传文件处理失败: {str(e)}")
                st.stop()
        
        st.stop()  # 如果找不到模型且没有上传，则停止应用
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}", exc_info=True)
        st.error(f"❌ 模型加载失败: {str(e)}")
        st.stop()

# ----------- 用户输入界面 -----------
def user_input_features():
    with st.sidebar:
        st.header("⚕️ Patient Parameters Input")
        
        # 使用两列布局优化空间
        col1, col2 = st.columns(2)
        
        with col1.expander("Hemodynamic Indicators", expanded=True):
              ef = st.slider('mean arterial pressure (mmHg)', 30, 180, 80,step=5.0,format="%d"
                         )
            cpb = st.number_input('temperature', 35, 43, 37, step=0.1,
                                format="%d")
            scr = st.number_input('Serum Creatinine (μmol/L)', 20.0, 500.0, 80.0, step=5.0,
                                format="%.1f", help="Renal function marker (normal: M 53-106, F 44-97)")
        
        with col2.expander("Other Parameters"):
              bl = st.number_input('age (year)', 18, 100, 50, step=5,
                               format="%d")
            gender = st.radio("Gender", ['Male', 'Female'], horizontal=True,
                            help="Biological sex")
            pwr = st.number_input('Platelet/WBC Ratio', 0.0, 50.0, 20.0, step=0.5,
                                format="%.1f", help="Inflammatory marker (normal range: 10-30)")
            tbil = st.number_input('Total Bilirubin (μmol/L)', 5.0, 300.0, 20.0, step=5.0,
                                 format="%.1f", help="Liver function marker (normal: 3.4-20.5)")

    return pd.DataFrame([[ef, cpb, scr, bl, 1 if gender == 'Male' else 0, pwr, tbil]],
                      columns=FEATURE_MAPPING.keys())

# ----------- SHAP解释可视化 -----------
def plot_shap_explanation(model, input_df):
    try:
        # 确保树模型使用TreeExplainer，线性模型用KernelExplainer
        if hasattr(model, 'tree_') or any(hasattr(model, est) for est in ['tree_', 'estimators_']):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(input_df, 10))
        
        # 计算SHAP值
        shap_values = explainer.shap_values(input_df)
        
        # 处理多分类/二分类
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # 假设第一类为负例（0），第二类为正例（肺炎风险）
            base_value = explainer.expected_value[1]
            shap_vals = shap_values[1]
        else:
            base_value = explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[0]
            shap_vals = shap_values
        
        # 创建可视化
        plt.figure(figsize=(10, 5))
        shap.force_plot(
            base_value=base_value,
            shap_values=shap_vals,
            features=input_df.values,
            feature_names=[FEATURE_MAPPING[c] for c in input_df.columns],
            matplotlib=True,
            show=False,
            text_rotation=15,
            plot_cmap=['#ff0051', '#008bfb']
        )
        
        plt.tight_layout()
        plt.gcf().set_facecolor('white')
        return plt.gcf()
    
    except Exception as e:
        logger.error(f"SHAP解释生成失败: {str(e)}", exc_info=True)
        st.error(f"特征解释生成失败: {str(e)}")
        return None

# ----------- 主界面 -----------
def main():
    st.title("Postoperative pneumonia after cardiac surgery")
    st.markdown("---")
    
    # 加载模型
    model = load_model()
    
    # 获取输入
    input_df = user_input_features()
    
    # 显示参数（使用漂亮的表格）
    with st.expander("📋 Current Input Parameters", expanded=True):
        # 创建漂亮的表格显示
        display_data = {
            "Parameter": [FEATURE_MAPPING[c] for c in input_df.columns],
            "Value": input_df.values.flatten().tolist()
        }
        st.dataframe(pd.DataFrame(display_data), use_container_width=True)
    
    # 预测按钮居中显示
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🚀 Start Risk Assessment", 
                               use_container_width=True, 
                               type="primary")
    
    # 预测结果展示
    if predict_btn:
        with st.spinner('Analyzing parameters...'):
            try:
                # 预测概率
                proba = model.predict_proba(input_df)[0][1]
                risk_percentage = f"{proba*100:.1f}%"
                risk_level = "High Risk" if proba > 0.5 else "Low Risk"
                color = "#FF4B4B" if proba > 0.5 else "#2ECC71"
                
                # 显示结果卡片
                st.markdown("---")
                st.subheader("Prediction Results")
                
                # 风险卡片（居中）
                _, center, _ = st.columns([1, 2, 1])
                with center:
                    st.markdown(f"""
                    <div style="border-radius: 15px; padding: 25px; background-color: {color}10; 
                                border-left: 8px solid {color}; margin: 20px 0; text-align: center;">
                        <h3 style="color: {color}; margin-top:0;">{risk_level}</h3>
                        <div style="font-size: 3rem; font-weight: bold; color: {color}; margin: 10px 0;">
                            {risk_percentage}
                        </div>
                        <div style="font-size: 1rem; color: #555;">
                            Pneumonia Probability (Threshold: 50%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 特征重要性分析
                st.subheader("📈 Feature Contribution Analysis")
                fig = plot_shap_explanation(model, input_df)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    st.caption("""
                    ​**Interpretation Guide**:
                    - → Red arrows indicate risk-increasing factors
                    - ← Blue arrows indicate risk-decreasing factors
                    - Arrow length represents impact magnitude
                    """)
                else:
                    st.warning("无法生成特征解释图。模型或SHAP可能不支持")
                
                # 临床建议
                st.markdown("---")
                st.subheader("🩺 Clinical Recommendations")
                if proba > 0.7:
                    st.warning("""
                    ​**🔴 High Risk Protocol**:
                    1. Enhanced respiratory monitoring - continuous pulse oximetry
                    2. Prophylactic antibiotics - consider Piperacillin-Tazobactam
                    3. Chest X-ray within 6 hours post-op
                    4. Arterial blood gas analysis every 4 hours
                    5. Consult pulmonologist immediately
                    """)
                elif proba > 0.5:
                    st.warning("""
                    ​**🟠 Moderate Risk Protocol**:
                    1. Incentive spirometry every 2 hours while awake
                    2. Daily serum procalcitonin levels
                    3. Strict fluid balance management (<1500mL/24hrs)
                    4. Pulmonary auscultation every 4 hours
                    5. Early mobilization protocol
                    """)
                else:
                    st.success("""
                    ​**🟢 Low Risk Protocol**:
                    1. Standard postoperative care
                    2. Maintain SpO₂ > 95% with supplemental O₂ as needed
                    3. Deep breathing exercises Q2H
                    4. Monitor for respiratory symptoms
                    5. Chest physiotherapy PRN
                    """)
                
                # 添加下载报告功能
                st.download_button(
                    label="📥 Download Clinical Report",
                    data=f"""
                    POSTOPERATIVE PNEUMONIA RISK ASSESSMENT REPORT\n
                    Patient Risk Level: {risk_level} ({risk_percentage})\n
                    Recommended Protocol: {"High Risk" if proba > 0.7 else "Moderate Risk" if proba > 0.5 else "Low Risk"}\n\n
                    INPUT PARAMETERS:\n
                    {pd.DataFrame({
                        "Parameter": [FEATURE_MAPPING[c] for c in input_df.columns],
                        "Value": input_df.values.flatten().tolist()
                    }).to_string(index=False)}
                    """,
                    file_name=f"VAP_Assessment_{risk_level.replace(' ', '_')}.txt",
                    mime="text/plain"
                )

            except Exception as e:
                logger.error(f"预测失败: {str(e)}", exc_info=True)
                st.error("预测错误 - 请检查输入参数或模型")
                st.info("技术细节错误:")
                st.code(str(e))

# ----------- 项目结构信息 -----------
def display_project_structure():
    """显示推荐的项目结构，帮助部署"""
    with st.expander("🏗️ Project Structure & Deployment Guide", expanded=False):
        st.write("""
        ​**Recommended GitHub Project Structure**:
        ```
        pneumonia-app/
        ├── models/
        │   └── my_model.pkl       # Your trained model
        ├── app.py                 # Main Streamlit application
        ├── requirements.txt       # Python dependencies
        └── README.md              # Project documentation
        ```
        
        ​**requirements.txt should include**:
        ```
        streamlit
        scikit-learn
        shap
        pandas
        matplotlib
        joblib
        """
        )
        
        st.write("""
        ​**For Streamlit Cloud Deployment**:
        1. Push your project to GitHub repository
        2. Go to [Streamlit Community Cloud](https://share.streamlit.io/)
        3. Click 'New App' and select your repository
        4. Set 'Main file path' to app.py
        5. Deploy!
        """)

# 主函数执行
if __name__ == '__main__':
    display_project_structure()  # 显示部署帮助信息
    main()
