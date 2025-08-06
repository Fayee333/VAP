# pneumonia_prediction_app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="机械通气患者误吸风险预测模型APP",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 特征名称映射
FEATURE_MAPPING = {
    'EF': 'Head of the bed elevated(°)',
    'CPB': 'Duration of mechanical ventilation (hours)',
    'SCr': 'APACHE II Score',
    'BL': 'age(year)',
    'Gender': 'Gastroesophageal Reflux Disease',
    'PWR': 'Length of stay in the ICU (days)',
    'TBIL': 'Glasgow Coma Scale (GCS) Score'
}

# ----------- 模型加载函数 -----------
@st.cache_resource
def load_model(file_path=None):
    """健壮的模型加载函数"""
    try:
        if file_path:
            return joblib.load(file_path)
        
        # 尝试多种可能的模型位置
        possible_paths = [
            Path("models") / "my_model.pkl",
            Path("my_model.pkl"),
            Path("app") / "models" / "my_model.pkl",
            Path("pneumonia_app") / "my_model.pkl",
            Path("resources") / "my_model.pkl"
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"加载模型: {path}")
                return joblib.load(path)
        
        logger.error(f"未找到模型文件。检查位置: {[str(p) for p in possible_paths]}")
        return None
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}", exc_info=True)
        st.error(f"❌ 模型加载失败: {str(e)}")
        return None

# ----------- 用户输入界面 -----------
def user_input_features():
    with st.sidebar:
        st.header("⚕️ Patient Parameters Input")
        
        # 使用两列布局优化空间
        col1, col2 = st.columns(2)
        
        with col1.expander("Hemodynamic Indicators", expanded=True):
            ef = st.slider('床头抬高（°）', 0, 45, 30, step=1)
            cpb = st.number_input('机械通气时间（小时）', 0, 480, 240, step=5)
            scr = st.number_input('APACHEII评分', min_value=0, max_value=71, value=20, step=1)
        
        with col2.expander("Other Parameters"):
            bl = st.number_input('年龄（岁）', min_value=18, max_value=100, value=50, step=5)
            gender = st.radio("胃食管反流疾病", ['是', '否'], horizontal=True)
            pwr = st.number_input('入住ICU时间（天）', min_value=0, max_value=50, value=20, step=1)
            tbil = st.number_input('GCS评分', min_value=0, max_value=15, value=7, step=1)

    return pd.DataFrame([[ef, cpb, scr, bl, 1 if gender == '是' else 0, pwr, tbil]],
                      columns=list(FEATURE_MAPPING.keys()))

# ----------- SHAP解释可视化 -----------
def plot_shap_explanation(model, input_df):
    try:
        if model is None:
            return None
            
        # 确保树模型使用TreeExplainer，线性模型用KernelExplainer
        if hasattr(model, 'tree_') or any(hasattr(model, est) for est in ['tree_', 'estimators_']):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(input_df, 10))
        
        # 计算SHAP值
        shap_values = explainer.shap_values(input_df)
        
        # 处理多分类/二分类
        if isinstance(shap_values, list) and len(shap_values) > 1:
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
    st.title("机械通气患者误吸风险预测模型APP")
    st.markdown("---")
    
    # 初始化session_state
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.uploaded_model = None
    
    # 模型加载选项
    if st.session_state.model is None:
        model_loaded = False
        
        # 显示模型上传选项
        st.subheader("模型加载选项")
        model_option = st.radio("选择模型来源:", 
                                ["自动加载预置模型", "上传自定义模型"])
        
        if model_option == "自动加载预置模型":
            st.session_state.model = load_model()
            model_loaded = st.session_state.model is not None
            if not model_loaded:
                st.warning("未找到预置模型，请尝试上传模型")
        else:
            uploaded_file = st.file_uploader("上传my_model.pkl文件", type=["pkl", "joblib"])
            if uploaded_file:
                try:
                    # 保存上传的模型
                    with st.spinner("加载上传模型中..."):
                        save_path = Path("uploaded_model.pkl")
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.session_state.model = joblib.load(save_path)
                        st.session_state.uploaded_model = save_path
                        model_loaded = True
                        st.success("模型上传并加载成功！")
                except Exception as e:
                    st.error(f"上传文件处理失败: {str(e)}")
    else:
        model_loaded = True
    
    # 如果模型已加载，显示输入界面
    if model_loaded:
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
                    proba = st.session_state.model.predict_proba(input_df)[0][1]
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
                    fig = plot_shap_explanation(st.session_state.model, input_df)
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
    
    # 项目结构信息
    display_project_structure()

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
    main()
