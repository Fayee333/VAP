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
    'EF': '床头抬高(°)',
    'CPB': '机械通气时间(小时)',
    'SCr': 'APACHE II评分',
    'BL': '年龄(岁)',
    'Gender': '胃食管反流疾病',
    'PWR': '入住ICU时间(天)',
    'TBIL': 'GCS评分'
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
        st.header("⚕️ 患者参数输入")
        
        # 使用两列布局优化空间
        col1, col2 = st.columns(2)
        
        with col1.expander("血流动力学指标", expanded=True):
            ef = st.slider('床头抬高(°)', 0, 45, 30, step=1)
            cpb = st.number_input('机械通气时间(小时)', 0, 480, 240, step=5)
            scr = st.number_input('APACHE II评分', min_value=0, max_value=71, value=20, step=1)
        
        with col2.expander("其他参数"):
            bl = st.number_input('年龄(岁)', min_value=18, max_value=100, value=50, step=5)
            gender = st.radio("胃食管反流疾病", ['是', '否'], horizontal=True)
            pwr = st.number_input('入住ICU时间(天)', min_value=0, max_value=50, value=20, step=1)
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
        with st.expander("📋 当前输入参数", expanded=True):
            # 创建漂亮的表格显示
            display_data = {
                "参数": [FEATURE_MAPPING[c] for c in input_df.columns],
                "值": input_df.values.flatten().tolist()
            }
            st.dataframe(pd.DataFrame(display_data), use_container_width=True)
        
        # 预测按钮居中显示
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_btn = st.button("🚀 开始风险评估", 
                                   use_container_width=True, 
                                   type="primary")
        
        # 预测结果展示
        if predict_btn:
            with st.spinner('分析参数中...'):
                try:
                    # 预测概率
                    proba = st.session_state.model.predict_proba(input_df)[0][1]
                    risk_percentage = f"{proba*100:.1f}%"
                    
                    # 三等级风险评估
                    if proba <= 0.4:
                        risk_level = "低风险"
                        color = "#2ECC71"  # 绿色
                        risk_description = "肺炎风险较低"
                    elif proba <= 0.7:
                        risk_level = "中风险"
                        color = "#FFC300"  # 黄色
                        risk_description = "肺炎风险中等"
                    else:
                        risk_level = "高风险"
                        color = "#FF4B4B"  # 红色
                        risk_description = "肺炎风险高"
                    
                    # 显示结果卡片
                    st.markdown("---")
                    st.subheader("预测结果")
                    
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
                                {risk_description}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 特征重要性分析
                    st.subheader("📈 特征贡献度分析")
                    fig = plot_shap_explanation(st.session_state.model, input_df)
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        st.caption("""
                        ​**解释说明**:
                        - → 红色箭头表示增加风险的因素
                        - ← 蓝色箭头表示降低风险的因素
                        - 箭头长度代表影响大小
                        """)
                    else:
                        st.warning("无法生成特征解释图。模型或SHAP可能不支持")
                    
                    # 临床建议
                    st.markdown("---")
                    st.subheader("🩺 临床建议")
                    if proba > 0.7:  # 高风险
                        st.error("""
                        ​**🔴 高风险处理方案**:
                        1. 加强呼吸监测 - 持续脉搏血氧监测
                        2. 预防性抗生素 - 考虑使用哌拉西林-他唑巴坦
                        3. 术后6小时内进行胸部X光检查
                        4. 每4小时进行动脉血气分析
                        5. 立即咨询呼吸科专家
                        """)
                    elif proba > 0.4:  # 中风险
                        st.warning("""
                        ​**🟠 中风险处理方案**:
                        1. 清醒状态下每2小时进行刺激性肺量测定
                        2. 每日检测血清降钙素原水平
                        3. 严格控制液体平衡(<1500mL/24小时)
                        4. 每4小时进行肺部听诊
                        5. 早期活动方案
                        """)
                    else:  # 低风险
                        st.success("""
                        ​**🟢 低风险处理方案**:
                        1. 标准术后护理
                        2. 通过补充氧气维持SpO₂ > 95%
                        3. 每2小时进行深呼吸练习
                        4. 监测呼吸症状
                        5. 按需进行胸部物理治疗
                        """)
                    
                    # 添加下载报告功能
                    st.download_button(
                        label="📥 下载临床报告",
                        data=f"""
                        肺炎风险评估报告\n
                        患者风险等级: {risk_level} ({risk_percentage})\n
                        推荐治疗方案: {"高风险" if proba > 0.7 else "中风险" if proba > 0.4 else "低风险"}\n\n
                        输入参数:\n
                        {pd.DataFrame({
                            "参数": [FEATURE_MAPPING[c] for c in input_df.columns],
                            "值": input_df.values.flatten().tolist()
                        }).to_string(index=False)}
                        """,
                        file_name=f"VAP_评估_{risk_level}.txt",
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
    with st.expander("🏗️ 项目结构与部署指南", expanded=False):
        st.write("""
        ​**推荐项目结构**:
        ```
        pneumonia-app/
        ├── models/
        │   └── my_model.pkl       # 训练好的模型
        ├── app.py                 # 主应用文件
        ├── requirements.txt       # Python依赖
        └── README.md              # 项目文档
        ```
        
        ​**requirements.txt 应包含**:
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
        ​**Streamlit云部署步骤**:
        1. 将项目推送到GitHub仓库
        2. 访问[Streamlit社区云](https://share.streamlit.io/)
        3. 点击"新建应用"并选择您的仓库
        4. 将"Main file path"设置为app.py
        5. 部署应用!
        """)

# 主函数执行
if __name__ == '__main__':
    main()
