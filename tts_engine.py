"""TTS 推理模块，封装模型复用、预热与错误分类。"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import multiprocessing as mp
import queue
from typing import Optional


@dataclass
class TTSRequest:
    text: str
    output_path: str
    am: str
    voc: str
    lang: str = "zh"
    spk_id: int = 0
    device: str = "cpu"
    speed: float = 1.0


class TTSInferenceError(RuntimeError):
    """用于向上层传递可读错误。"""


class TTSEngine:
    def __init__(self) -> None:
        self._executor = None
        self._speed_param_name: Optional[str] = None

    def warmup(self, device: str = "cpu") -> None:
        # 用极短文本触发模型加载与缓存准备。
        self.synthesize(
            TTSRequest(
                text="你好",
                output_path="__warmup__.wav",
                am="fastspeech2_csmsc",
                voc="hifigan_csmsc",
                device=device,
            ),
            remove_output=True,
        )

    def synthesize(self, req: TTSRequest, remove_output: bool = False) -> None:
        import os

        executor = self._get_executor()
        infer_kwargs = {
            "text": req.text,
            "output": req.output_path,
            "am": req.am,
            "voc": req.voc,
            "lang": req.lang,
            "spk_id": req.spk_id,
            "device": req.device,
        }
        speed_param = self.get_speed_param_name()
        if req.speed != 1.0 and speed_param:
            infer_kwargs[speed_param] = req.speed
        try:
            executor(**infer_kwargs)
        except Exception as exc:
            raise TTSInferenceError(self._classify_error(exc)) from exc
        finally:
            if remove_output and os.path.exists(req.output_path):
                os.remove(req.output_path)

    def _get_executor(self):
        if self._executor is None:
            from paddlespeech.cli.tts import TTSExecutor

            self._executor = TTSExecutor()
        return self._executor

    def get_speed_param_name(self) -> Optional[str]:
        if self._speed_param_name is not None:
            return self._speed_param_name
        executor = self._get_executor()
        sig = inspect.signature(executor.__call__)
        candidates = ("speed", "am_speed", "speed_degree")
        for name in candidates:
            if name in sig.parameters:
                self._speed_param_name = name
                return name
        self._speed_param_name = ""
        return None

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        msg = str(exc).strip()
        low = msg.lower()
        if "cuda" in low or "cudnn" in low or "gpu" in low:
            return "GPU 不可用或驱动不匹配，请切换 CPU 或检查 CUDA 环境。"
        if "no module named" in low:
            return "缺少依赖，请确认在正确虚拟环境中安装 requirements。"
        if "permission" in low:
            return "文件权限不足，请检查输出目录权限。"
        if "no such file" in low:
            return "模型或输出路径不存在，请检查配置与目录。"
        return msg or "未知推理错误"


def _isolated_worker(req: TTSRequest, out_q: mp.Queue):
    try:
        engine = TTSEngine()
        engine.synthesize(req)
        out_q.put(("ok", ""))
    except Exception as exc:  # pragma: no cover
        out_q.put(("err", str(exc)))


def synthesize_with_hard_timeout(req: TTSRequest, timeout_sec: int) -> None:
    out_q: mp.Queue = mp.Queue()
    proc = mp.Process(target=_isolated_worker, args=(req, out_q), daemon=True)
    proc.start()
    proc.join(timeout=timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        raise TimeoutError(f"语音合成超时（>{timeout_sec}s）并已终止任务。")

    try:
        status, message = out_q.get_nowait()
    except queue.Empty:
        if proc.exitcode == 0:
            return
        raise TTSInferenceError("推理进程异常退出。")

    if status == "err":
        raise TTSInferenceError(message)
