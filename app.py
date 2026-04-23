"""
语音合成 Web 应用
基于百度飞桨 FastSpeech2 + 可选声码器
功能：中文文本合成、音色选择、声码器选择、语速调节、合成历史记录（文件持久化，支持单条删除）
"""
import subprocess
import sys

# 在代码运行时强制安装兼容的 SDK 版本
def install_compat_sdk():
    try:
        import aistudio_sdk
    except ImportError:
        # 使用 pip 强制安装特定旧版本以解决导入错误
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aistudio-sdk==0.0.1"])

install_compat_sdk()

import streamlit as st

import librosa
import soundfile as sf
import os
import tempfile
import time
from datetime import datetime
import copy
from pathlib import Path
import concurrent.futures
import uuid

from history_store import HistoryStore, HISTORY_SCHEMA_VERSION
from preprocess import preprocess_for_tts
from tts_engine import TTSEngine, TTSRequest, TTSInferenceError, synthesize_with_hard_timeout

# 性能优化（可用环境变量覆盖）
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
os.environ["FLAGS_use_mkldnn"] = os.environ.get("FLAGS_use_mkldnn", "1")

# ========== 持久化文件路径 ==========
HISTORY_FILE = "tts_history.json"
OUTPUT_DIR = Path("outputs")
TTS_TIMEOUT_SEC = 180
MAX_ERROR_CHARS = 4000
TTS_DEVICE = os.environ.get("PADDLESPEECH_DEVICE", "cpu")
_TTS_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)
store = HistoryStore(Path(HISTORY_FILE), OUTPUT_DIR)


@st.cache_resource(show_spinner=False)
def get_tts_engine():
    return TTSEngine()


@st.cache_data(show_spinner=False)
def model_speed_param_name() -> str:
    try:
        return get_tts_engine().get_speed_param_name() or ""
    except Exception:
        return ""

st.set_page_config(page_title="智音中文语音合成助手", layout="centered")

if 'history' not in st.session_state:
    st.session_state.history = store.load()
if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None
if 'current_meta' not in st.session_state:
    st.session_state.current_meta = {}
if "tts_running" not in st.session_state:
    st.session_state.tts_running = False

def add_to_history(meta):
    st.session_state.history = store.append(st.session_state.history, meta)

def delete_history_item(index):
    st.session_state.history = store.remove(st.session_state.history, index)

def clear_history():
    st.session_state.history = store.clear()

def make_delete_callback(idx):
    def callback():
        delete_history_item(idx)
        # 自动 rerun，无需显式调用
    return callback

def save_current_to_history():
    if st.session_state.current_meta and st.session_state.current_audio:
        meta_copy = copy.deepcopy(st.session_state.current_meta)
        add_to_history(meta_copy)
        st.success("已保存到历史记录！")
    else:
        st.warning("没有可保存的音频")


def build_output_filename(voice: str, speed: float) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    voice_slug = voice.replace(" ", "_").replace("(", "").replace(")", "")
    return OUTPUT_DIR / f"tts_{voice_slug}_{speed:.1f}x_{ts}_{suffix}.wav"


def synthesize_audio(req: TTSRequest, use_isolated_mode: bool, timeout_sec: int) -> None:
    if use_isolated_mode:
        synthesize_with_hard_timeout(req=req, timeout_sec=timeout_sec)
        return

    engine = get_tts_engine()
    future = _TTS_THREAD_POOL.submit(engine.synthesize, req)
    try:
        future.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError as exc:
        raise TimeoutError(f"语音合成超时（>{timeout_sec}s）。") from exc

st.title(" 智音中文语音合成助手")
st.markdown("基于深度学习 FastSpeech2 + 可选声码器，支持语速调节和音色选择")

with st.expander(" 使用说明", expanded=False):
    st.markdown("""
    1. 输入中文文本（数字自动转中文，英文自动过滤）
    2. 选择音色和声码器（HiFiGAN 音质更好，MB-MelGAN 速度更快）
    3. 调节语速滑块
    4. 点击「开始合成」
    5. 试听后可将音频保存到历史记录，或直接下载
    """)

AM_OPTIONS = {
    "标准女声 (CSMSC)": "fastspeech2_csmsc",
    "男声": "fastspeech2_male",
    "粤语女声": "fastspeech2_canton",
}
CANTON_EXTRA = {"lang": "canton", "spk_id": 10}

st.subheader("合成参数设置")
col1, col2, col3 = st.columns(3)
with col1:
    voice_choice = st.selectbox("音色", list(AM_OPTIONS.keys()))
with col2:
    voc_display = st.selectbox("声码器", ["HiFiGAN (高质量)", "MB-MelGAN (快速)"])
with col3:
    speed = st.slider("语速", 0.5, 2.0, 1.0, 0.1)

voc_model = "hifigan_csmsc" if voc_display == "HiFiGAN (高质量)" else "mb_melgan_csmsc"
if voice_choice == "男声" and voc_display == "HiFiGAN (高质量)":
    voc_model = "hifigan_male"
if voice_choice == "粤语女声":
    voc_model = "pwgan_aishell3"
    st.info("粤语模式自动使用 pwgan_aishell3 声码器")

with st.sidebar:
    st.header("运行设置")
    isolated_mode = st.toggle("稳定模式（隔离进程，支持硬超时）", value=False)
    timeout_sec = st.slider("合成超时（秒）", min_value=30, max_value=600, value=TTS_TIMEOUT_SEC, step=10)
    if st.button("模型预热（首次建议执行）", use_container_width=True):
        try:
            with st.spinner("正在预热模型..."):
                get_tts_engine().warmup(device=TTS_DEVICE)
            st.success("预热完成，后续首次合成会更快。")
        except Exception as e:
            st.error(f"预热失败：{str(e)[-MAX_ERROR_CHARS:]}")

    st.header("合成历史")
    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history):
            with st.expander(f"{item.get('time', '未知')} - {item.get('text', '')[:15]}...", expanded=False):
                st.caption(f"**原文**: {item.get('original_text', '')}")
                st.caption(f"**音色**: {item.get('voice', '')}  ·  **语速**: {item.get('speed', 1.0)}x")
                st.caption(f"**版本**: v{item.get('schema_version', HISTORY_SCHEMA_VERSION)}")
                audio_path = item.get('audio_path', '')
                abs_path = store.resolve_audio_path(audio_path) if audio_path else None
                if abs_path and abs_path.exists():
                    st.audio(str(abs_path), format='audio/wav')
                else:
                    st.caption("⚠️ 音频文件已不存在")
                st.button("🗑️ 删除此条", key=f"del_{idx}", on_click=make_delete_callback(idx))
        if st.button("🧹 清空所有历史"):
            clear_history()
            st.rerun()
    else:
        st.caption("暂无合成记录")

text_input = st.text_area(
    "文本编辑区",
    value="请输入要合成的文本",
    height=120,
    max_chars=1000
)

if text_input:
    processed = preprocess_for_tts(text_input)
    if processed != text_input:
        with st.expander("查看文本预处理结果"):
            st.caption(f"原文：{text_input}")
            st.caption(f"处理后：{processed}")

if speed != 1.0:
    if model_speed_param_name():
        st.caption("提示：当前优先使用模型侧调速，速度更快且音质更稳定。")
    else:
        st.caption("提示：当前使用后处理改速，会增加耗时并可能轻微影响音质。")

if st.button("开始合成", type="primary", use_container_width=True, disabled=st.session_state.tts_running):
    if not text_input.strip():
        st.error("❌ 请输入要合成的文本！")
    elif st.session_state.tts_running:
        st.warning("当前已有合成任务正在执行，请稍候。")
    else:
        st.session_state.tts_running = True
        progress_bar = st.progress(0, "准备中...")
        status_text = st.empty()
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            status_text.text("📝 文本预处理中...")
            progress_bar.progress(10)
            clean_text = preprocess_for_tts(text_input)

            output_file = build_output_filename(voice_choice, speed)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name

            status_text.text("🎙️ 声学模型推理中...")
            progress_bar.progress(30)

            am_model = AM_OPTIONS[voice_choice]
            lang = CANTON_EXTRA["lang"] if voice_choice == "粤语女声" else "zh"
            spk_id = CANTON_EXTRA["spk_id"] if voice_choice == "粤语女声" else 0
            req = TTSRequest(
                text=clean_text,
                output_path=temp_path,
                am=am_model,
                voc=voc_model,
                lang=lang,
                spk_id=spk_id,
                device=TTS_DEVICE,
                speed=speed,
            )
            synthesize_audio(req, use_isolated_mode=isolated_mode, timeout_sec=timeout_sec)

            status_text.text("🔊 生成音频波形中...")
            progress_bar.progress(60)

            status_text.text("🎵 调节语速中...")
            progress_bar.progress(80)
            y, sr = librosa.load(temp_path, sr=None)
            if speed != 1.0 and not model_speed_param_name():
                y = librosa.effects.time_stretch(y, rate=speed)
            sf.write(str(output_file), y, sr)

            progress_bar.progress(100)
            status_text.text("✅ 合成完成！")

            st.session_state.current_audio = str(output_file)
            st.session_state.current_meta = {
                'time': datetime.now().strftime('%H:%M:%S'),
                'created_at': datetime.now().isoformat(timespec="seconds"),
                'schema_version': HISTORY_SCHEMA_VERSION,
                'original_text': text_input,
                'text': clean_text,
                'voice': voice_choice,
                'speed': speed,
                'audio_path': str(output_file.relative_to(OUTPUT_DIR.parent)),
                'tts_device': TTS_DEVICE,
                'isolated_mode': isolated_mode,
                'timeout_sec': timeout_sec,
                'am': am_model,
                'voc': voc_model,
                'model_speed_param': model_speed_param_name() or "",
            }

            st.success(f"✅ 合成成功！音色：{voice_choice}，语速：{speed}x")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

        except TimeoutError as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ {str(e)}")
        except TTSInferenceError as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ 语音合成失败：{str(e)[-MAX_ERROR_CHARS:]}")
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ 发生错误：{str(e)}")
        finally:
            st.session_state.tts_running = False
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)

if st.session_state.current_audio and os.path.exists(st.session_state.current_audio):
    st.markdown("### 当前合成结果")
    meta = st.session_state.current_meta
    st.caption(f"文本：{meta.get('original_text', '')}")
    st.caption(f"音色：{meta.get('voice', '')}  ·  语速：{meta.get('speed', 1.0)}x")
    st.audio(st.session_state.current_audio, format="audio/wav")

    col1, col2 = st.columns(2)
    with col1:
        st.button("保存到历史记录", on_click=save_current_to_history, use_container_width=True)
    with col2:
        with open(st.session_state.current_audio, "rb") as f:
            audio_bytes = f.read()
        st.download_button(
            label="下载音频",
            data=audio_bytes,
            file_name=f"tts_{meta.get('voice', 'unknown').replace(' ', '_')}_{meta.get('speed', 1.0)}x.wav",
            mime="audio/wav",
            use_container_width=True
        )

st.markdown("---")
st.markdown("💡 提示：文本越长，合成时间越长，请耐心等待。")