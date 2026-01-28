import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import time
import io
import math

# ==========================================
# 1. 页面配置与 UI 美化 (UI/UX)
# ==========================================
st.set_page_config(
    page_title="AI 智能数据清洗工坊",
    page_icon="🧼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS - 现代极简 Teal 风格
st.markdown("""
<style>
    /* 全局字体与背景 */
    .stApp {
        background-color: #F8F9FA;
        font-family: 'Inter', sans-serif;
    }

    /* 标题样式 */
    h1, h2, h3 {
        color: #2C3E50;
        font-weight: 600;
    }

    /* 按钮样式 - 渐变青色 */
    .stButton>button {
        background: linear-gradient(135deg, #20B2AA 0%, #008080 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 128, 128, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 128, 128, 0.3);
        color: white;
    }

    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }

    /* 卡片式容器效果 */
    .stTextArea, .stSelectbox, .stFileUploader {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
    }

    /* 进度条颜色 */
    .stProgress > div > div > div > div {
        background-color: #20B2AA;
    }

    /* 成功消息 */
    .stSuccess {
        background-color: #E0F2F1;
        border-left: 4px solid #008080;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Session State 初始化 (状态管理)
# ==========================================
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_result' not in st.session_state:
    st.session_state.df_result = None
if 'optimized_prompt' not in st.session_state:
    st.session_state.optimized_prompt = ""
if 'unique_values' not in st.session_state:
    st.session_state.unique_values = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# ==========================================
# 3. 侧边栏配置 (Sidebar)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/cleaning-service.png", width=100)
    st.title("配置中心")

    st.markdown("### 🔑 API 设置")
    api_key = st.text_input("DeepSeek API Key", type="password", placeholder="sk-...",
                            help="请输入您的 DeepSeek API Key")
    base_url = st.text_input("Base URL", value="https://api.deepseek.com", help="默认为 DeepSeek 官方地址")

    st.markdown("### 🧠 模型选择")
    model_option = st.selectbox("选择模型", ["deepseek-chat", "deepseek-coder"], index=0)

    st.markdown("---")
    st.info("""
    **关于本工具**

    这是一个基于 AI 的通用数据清洗助手。
    1. 上传 Excel/CSV
    2. 告诉 AI 你想做什么
    3. 自动批量处理并导出
    """)


# ==========================================
# 4. 核心逻辑函数
# ==========================================

def get_client():
    if not api_key:
        st.error("请先在侧边栏输入 API Key")
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


def ai_optimize_prompt(user_intent):
    """使用 AI 将用户意图转化为结构化 System Prompt"""
    client = get_client()
    if not client: return None

    meta_prompt = f"""
    你是一名资深的 Prompt Engineer。你的任务是将用户的【简单数据处理意图】转化为一段【专业的、生产级的 System Prompt】。

    用户意图: "{user_intent}"

    请生成的 System Prompt 必须包含以下要求：
    1. 设定明确的角色（如医学专家、地理学家等）。
    2. 定义具体的分类标准或提取规则。
    3. **强制要求**输出格式为严格的 JSON，格式如下：
       {{"results": [{{"text": "原始文本", "code": "分类代码/结果", "note": "简短理由"}}]}}
    4. 要求思维链（CoT），但在 JSON 中只返回最终结果。
    5. 语气要严肃、准确。

    请直接返回生成的 System Prompt 内容，不要包含任何解释性文字或 markdown 代码块标记。
    """

    try:
        with st.spinner("✨ AI 正在思考最佳指令..."):
            response = client.chat.completions.create(
                model=model_option,
                messages=[{"role": "user", "content": meta_prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
    except Exception as e:
        st.error(f"优化失败: {e}")
        return None


def process_batch(client, batch_texts, system_prompt, max_retries=3):
    """批量处理核心函数，包含重试机制"""
    user_prompt = f"待处理列表: {json.dumps(batch_texts, ensure_ascii=False)}"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_option,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            content = response.choices[0].message.content
            # 清洗 markdown 标记
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            return json.loads(content)

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:  # Rate Limit
                wait_time = 2 ** (attempt + 1)  # Exponential backoff
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                return {}  # Failed after retries
            else:
                time.sleep(1)
    return {}


# ==========================================
# 5. 主界面布局
# ==========================================

st.title("🧼 AI 智能数据清洗工坊")
st.markdown("通过自然语言指令，利用 DeepSeek 模型批量清洗、分类或提取 Excel 数据。")

# --- 步骤 1: 数据加载 ---
st.header("1. 数据加载与选择")
uploaded_file = st.file_uploader("上传 Excel 或 CSV 文件", type=['xlsx', 'csv'])

if uploaded_file:
    # 读取文件
    if st.session_state.df_raw is None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df_raw = df
        except Exception as e:
            st.error(f"文件读取失败: {e}")

    if st.session_state.df_raw is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.caption("数据预览 (前5行)")
            st.dataframe(st.session_state.df_raw.head(), use_container_width=True, height=200)

        with col2:
            st.info("👇 请选择需要清洗/分析的目标列")
            target_cols = st.multiselect(
                "目标列 (Target Columns)",
                options=st.session_state.df_raw.columns.tolist()
            )

        # 提取唯一值逻辑
        if target_cols:
            unique_items = set()
            for col in target_cols:
                # 简单清洗：转字符串，去首尾空格，排除空值
                items = st.session_state.df_raw[col].dropna().astype(str).apply(lambda x: x.strip()).unique()
                items = [t for t in items if t != '' and t.lower() != 'nan']
                unique_items.update(items)

            st.session_state.unique_values = list(unique_items)
            st.success(f"🔍 扫描完成：共发现 {len(st.session_state.unique_values)} 个唯一值 (Unique Values) 待处理。")

# --- 步骤 2: 智能指令构建 ---
if st.session_state.df_raw is not None and st.session_state.unique_values:
    st.markdown("---")
    st.header("2. AI 提示词 (Prompt) 优化")

    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            user_intent = st.text_input(
                "💬 用“人话”描述你的任务",
                placeholder="例如：判断这些文本是不是生殖系统疾病，或者提取地址中的省份...",
                help="输入你想让 AI 做的事情，点击优化按钮生成专业指令。"
            )
        with c2:
            st.write("")  # Spacer
            st.write("")
            optimize_btn = st.button("✨ 帮我优化提示词", use_container_width=True)

        if optimize_btn and user_intent:
            optimized = ai_optimize_prompt(user_intent)
            if optimized:
                st.session_state.optimized_prompt = optimized

        if st.session_state.optimized_prompt:
            st.markdown("##### 📝 最终使用的 System Prompt (可手动微调)")
            final_prompt = st.text_area(
                "System Prompt",
                value=st.session_state.optimized_prompt,
                height=250,
                help="这是实际发送给模型的指令，您可以手动修改 JSON 结构要求或其他规则。"
            )
            st.session_state.optimized_prompt = final_prompt

# --- 步骤 3: 批量处理 ---
if st.session_state.optimized_prompt and st.session_state.unique_values:
    st.markdown("---")
    st.header("3. 批量处理执行")

    col_run, col_status = st.columns([1, 4])
    with col_run:
        start_btn = st.button("🚀 开始处理", type="primary", use_container_width=True)

    if start_btn:
        client = get_client()
        if client:
            results_map = {}
            batch_size = 30  # 可根据 Token 长度调整
            total_items = st.session_state.unique_values
            total_batches = math.ceil(len(total_items) / batch_size)

            # 初始化进度条
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(total_items))
                batch = total_items[start_idx:end_idx]

                status_text.markdown(f"**正在处理第 {i + 1}/{total_batches} 批** ({start_idx}-{end_idx} 条)...")

                # 调用 AI
                api_res = process_batch(client, batch, st.session_state.optimized_prompt)

                # 解析结果
                if "results" in api_res:
                    for item in api_res["results"]:
                        # 健壮性检查：确保 key 存在
                        text_key = item.get("text")
                        if text_key:
                            results_map[text_key] = {
                                "code": item.get("code", "N/A"),
                                "note": item.get("note", "")
                            }

                # 更新进度
                progress_bar.progress((i + 1) / total_batches)

            status_text.success("✅ 所有批次处理完成！")
            st.balloons()

            # --- 结果回填逻辑 ---
            df_result = st.session_state.df_raw.copy()

            # 创建映射字典
            code_map = {k: v['code'] for k, v in results_map.items()}
            note_map = {k: v['note'] for k, v in results_map.items()}

            # 回填到 DataFrame
            target_cols = [col for col in df_result.columns if df_result[col].isin(
                results_map.keys()).any() or col in st.session_state.df_raw.columns]  # 简化逻辑，重新获取选中的列比较复杂，这里假设用户没改选

            # 这里我们需要重新获取用户步骤1选中的列，由于 Streamlit 刷新机制，
            # 最好是在步骤1就存入 session_state。为简化，我们假设用户现在选的列就是刚才处理的列。
            # 实际应用中可以锁定 session_state['selected_cols']

            for col in df_result.columns:
                # 检查该列是否有值在我们的 results_map 中
                sample_values = df_result[col].astype(str).head(20).tolist()
                is_target = any([v.strip() in results_map for v in sample_values])

                if is_target:
                    # 插入列
                    col_idx = df_result.columns.get_loc(col)

                    mapped_notes = df_result[col].astype(str).str.strip().map(note_map).fillna("")
                    mapped_codes = df_result[col].astype(str).str.strip().map(code_map).fillna("")

                    df_result.insert(col_idx + 1, f"{col}_AI说明", mapped_notes)
                    df_result.insert(col_idx + 1, f"{col}_AI结果", mapped_codes)

            st.session_state.df_result = df_result
            st.session_state.processing_complete = True

# --- 步骤 4: 结果预览与导出 ---
if st.session_state.processing_complete and st.session_state.df_result is not None:
    st.markdown("---")
    st.header("4. 结果下载")

    st.dataframe(st.session_state.df_result.head(), use_container_width=True)

    # 转换为 Excel 字节流
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        st.session_state.df_result.to_excel(writer, index=False)
    processed_data = output.getvalue()

    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.download_button(
            label="📥 下载处理后的 Excel",
            data=processed_data,
            file_name="AI_Cleaned_Data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
    with col_d2:
        st.metric(label="处理总行数", value=len(st.session_state.df_result))