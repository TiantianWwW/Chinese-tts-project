# tts_project（Streamlit + PaddleSpeech）

这是一个基于 **Streamlit** 的中文语音合成小工具，通过 **PaddleSpeech 的 Python 推理接口（进程内常驻模型）** 完成 TTS 合成，并提供：

- 文本预处理（统一走 `preprocess.py` 管道实现）
- 音色选择（标准女声 / 男声 / 粤语女声）
- 声码器选择（HiFiGAN / MB-MelGAN；粤语自动使用 `pwgan_aishell3`）
- 语速调节（优先模型侧调速，不支持时自动回退后处理）
- 合成历史（`tts_history.json`，支持删除，带 schema 版本）
- 运行模式切换（性能模式 / 稳定模式）

## 环境要求

- Windows 10/11
- Python（建议使用虚拟环境）
- 依赖见 `requirements-app.txt`（推荐）

## 安装

在项目目录下执行：

推荐安装运行时依赖：

```bash
pip install -r requirements-app.txt
```

如果你需要跑测试：

```bash
pip install -r requirements-dev.txt
```

## 启动

```bash
streamlit run app.py
```

启动后浏览器会打开页面。

## 输出文件与历史

- 合成音频默认输出到 `outputs/` 目录
- 历史记录保存在 `tts_history.json`
- 历史中的 `audio_path` 使用相对路径，项目迁移后更容易复用

## 运行模式

- **性能模式（默认）**
  - 进程内复用已加载模型，吞吐更好，适合常规使用。
- **稳定模式（侧边栏开关）**
  - 每次合成在隔离子进程运行，支持“硬超时终止”，更适合虚拟机/不稳定环境。

## 常见问题

- **首次合成很慢**
  - 首次会下载/初始化模型与缓存；之后会复用同一进程内已加载的模型，速度会明显变快。

- **合成超时**
  - 可能是文本过长、首次下载模型/缓存、或机器性能不足。可以先缩短文本重试。
  - 稳定模式下超时会直接终止任务；性能模式下属于软超时提示。

- **语速调节为什么有时变慢**
  - 若当前模型不支持模型侧调速，会自动回退到后处理改速（`librosa`），这会更耗时。

- **切换 CPU/GPU**
  - 默认使用 CPU。你可以通过环境变量设置设备：`PADDLESPEECH_DEVICE=cpu` 或 `PADDLESPEECH_DEVICE=gpu`，再启动 `streamlit run app.py`。

## 开发与测试

```bash
python -m unittest discover -s tests
```

