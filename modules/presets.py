# -*- coding:utf-8 -*-
import os
from pathlib import Path
import gradio as gr
from .webui_locale import I18nAuto

i18n = I18nAuto()  # internationalization

CHATGLM_MODEL = None
CHATGLM_TOKENIZER = None
LLAMA_MODEL = None
LLAMA_INFERENCER = None

# ChatGPT 设置
INITIAL_SYSTEM_PROMPT = "You are a helpful assistant."
API_HOST = "openai.api2d.net"
COMPLETION_URL = f"https://{API_HOST}/v1/chat/completions"
BALANCE_API_URL = f"https://{API_HOST}/dashboard/billing/credit_grants"
USAGE_API_URL = f"https://{API_HOST}/dashboard/billing/usage"
CHATGLM_6B_URL = "http://10.109.68.1:8081/v1/chat/completions"
HISTORY_DIR = Path("history")
HISTORY_DIR = "history"
TEMPLATES_DIR = "templates"

# 错误信息
STANDARD_ERROR_MSG = i18n("☹️发生了错误：")  # 错误信息的标准前缀
GENERAL_ERROR_MSG = i18n("获取对话时发生错误，请查看后台日志")
ERROR_RETRIEVE_MSG = i18n("请检查网络连接，或者API-Key是否有效。")
CONNECTION_TIMEOUT_MSG = i18n("连接超时，无法获取对话。")  # 连接超时
READ_TIMEOUT_MSG = i18n("读取超时，无法获取对话。")  # 读取超时
PROXY_ERROR_MSG = i18n("代理错误，无法获取对话。")  # 代理错误
SSL_ERROR_PROMPT = i18n("SSL错误，无法获取对话。")  # SSL 错误
NO_APIKEY_MSG = i18n("API key为空，请检查是否输入正确。")  # API key 长度不足 51 位
NO_INPUT_MSG = i18n("请输入对话内容。")  # 未输入对话内容
BILLING_NOT_APPLICABLE_MSG = i18n("账单信息不适用") # 本地运行的模型返回的账单信息

TIMEOUT_STREAMING = 60  # 流式对话时的超时时间
TIMEOUT_ALL = 200  # 非流式对话时的超时时间
ENABLE_STREAMING_OPTION = True  # 是否启用选择选择是否实时显示回答的勾选框
HIDE_MY_KEY = False  # 如果你想在UI中隐藏你的 API 密钥，将此值设置为 True
CONCURRENT_COUNT = 100 # 允许同时使用的用户数量

SIM_K = 5
INDEX_QUERY_TEMPRATURE = 1.0

CHUANHU_TITLE = i18n("ChatEDU 🚀")
BACK_LINK = """其他链接： <a href="http://10.101.120.11:8000" target="_blank">个性化学习Demo</a>  &nbsp;&nbsp;&nbsp;
 <a href="http://10.109.68.1:7866" target="_blank"> 图像生成Demo </>"""
FOOTER = """<div class="versions">{versions}</div>"""

APPEARANCE_SWITCHER = """
<div style="display: flex; justify-content: space-between;">
<span style="margin-top: 4px !important;">"""+ i18n("切换亮暗色主题")  + """</span>
<span><label class="apSwitch" for="checkbox">
    <input type="checkbox" id="checkbox">
    <div class="apSlider"></div>
</label></span>
</div>
"""

SUMMARIZE_PROMPT = "你是谁？我们刚才聊了什么？"  # 总结对话时的 prompt

ONLINE_MODELS = [
    "chatglm-130b",
    "gpt-3.5-turbo",
    # "xmchat"
]

LOCAL_MODELS = [
    "chatglm-6b",
    # "StableLM",
    "MOSS"
]

if os.environ.get('HIDE_LOCAL_MODELS', 'false') == 'true':
    MODELS = ONLINE_MODELS
else:
    MODELS = ONLINE_MODELS + LOCAL_MODELS

DEFAULT_MODEL = 0

os.makedirs("models", exist_ok=True)
os.makedirs("lora", exist_ok=True)
os.makedirs("history", exist_ok=True)
for dir_name in os.listdir("models"):
    if os.path.isdir(os.path.join("models", dir_name)):
        if dir_name not in MODELS:
            MODELS.append(dir_name)

MODEL_TOKEN_LIMIT = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "chatglm-130b": 4096
}

TOKEN_OFFSET = 1000 # 模型的token上限减去这个值，得到软上限。到达软上限之后，自动尝试减少token占用。
DEFAULT_TOKEN_LIMIT = 3000 # 默认的token上限
REDUCE_TOKEN_FACTOR = 0.5 # 与模型token上限想乘，得到目标token数。减少token占用时，将token占用减少到目标token数以下。

REPLY_LANGUAGES = [
    "简体中文",
    "繁體中文",
    "English",
    "日本語",
    "Español",
    "Français",
    "Deutsch",
    "跟随问题语言（不稳定）"
]


WEBSEARCH_PTOMPT_TEMPLATE = """\
Web search results:

{web_results}
Current date: {current_date}

Instructions: Using the provided web search results, write a comprehensive reply to the given query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.
Query: {query}
Reply in {reply_language}
"""

PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context_str}
---------------------
Current date: {current_date}.
Using the provided context information, write a comprehensive reply to the given query.
Make sure to cite results using [number] notation after the reference.
If the provided context information refer to multiple subjects with the same name, write separate answers for each subject.
Use prior knowledge only if the given context didn't provide enough information.
Answer the question: {query_str}
Reply in {reply_language}
"""

REFINE_TEMPLATE = """\
The original question is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer
(only if needed) with some more context below.
------------
{context_msg}
------------
Given the new context, refine the original answer to better
Reply in {reply_language}
If the context isn't useful, return the original answer.
"""

ALREADY_CONVERTED_MARK = "<!-- ALREADY CONVERTED BY PARSER. -->"

small_and_beautiful_theme = gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#02C160",
            c100="rgba(2, 193, 96, 0.2)",
            c200="#02C160",
            c300="rgba(2, 193, 96, 0.32)",
            c400="rgba(2, 193, 96, 0.32)",
            c500="rgba(2, 193, 96, 1.0)",
            c600="rgba(2, 193, 96, 1.0)",
            c700="rgba(2, 193, 96, 0.32)",
            c800="rgba(2, 193, 96, 0.32)",
            c900="#02C160",
            c950="#02C160",
        ),
        secondary_hue=gr.themes.Color(
            c50="#576b95",
            c100="#576b95",
            c200="#576b95",
            c300="#576b95",
            c400="#576b95",
            c500="#576b95",
            c600="#576b95",
            c700="#576b95",
            c800="#576b95",
            c900="#576b95",
            c950="#576b95",
        ),
        neutral_hue=gr.themes.Color(
            name="gray",
            c50="#f9fafb",
            c100="#f3f4f6",
            c200="#e5e7eb",
            c300="#d1d5db",
            c400="#B2B2B2",
            c500="#808080",
            c600="#636363",
            c700="#515151",
            c800="#393939",
            c900="#272727",
            c950="#171717",
        ),
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        button_primary_background_fill="#06AE56",
        button_primary_background_fill_dark="#06AE56",
        button_primary_background_fill_hover="#07C863",
        button_primary_border_color="#06AE56",
        button_primary_border_color_dark="#06AE56",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
        button_secondary_background_fill="#F2F2F2",
        button_secondary_background_fill_dark="#2B2B2B",
        button_secondary_text_color="#393939",
        button_secondary_text_color_dark="#FFFFFF",
        # background_fill_primary="#F7F7F7",
        # background_fill_primary_dark="#1F1F1F",
        block_title_text_color="*primary_500",
        block_title_background_fill="*primary_100",
        input_background_fill="#F6F6F6",
    )
