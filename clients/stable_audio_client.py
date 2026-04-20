"""Stable Audio 2.5 API Client - 完整实现，符合官方 OpenAPI 规范"""

import asyncio
import requests
from requests_toolbelt import MultipartEncoder
from pathlib import Path
from typing import Optional

from config import config
from utils.logger import setup_logger

logger = setup_logger("StableAudio")


class StableAudioClient:
    def __init__(self):
        self.api_key = config.api.stability_api_key
        if not self.api_key:
            logger.warning("STABILITY_API_KEY not set. Stable Audio disabled.")
            self.available = False
        else:
            self.available = True
        self.base_url = "https://api.stability.ai"
        self.max_duration = 190  # 官方最大时长

    async def generate(
        self,
        prompt: str,
        duration: int = 30,
        output_path: Optional[Path] = None,
        model: str = "stable-audio-2.5",
        output_format: str = "wav",
        seed: int = 0,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
    ) -> Optional[Path]:
        """
        文本生成音频
        :param prompt: 音频描述文本
        :param duration: 时长（秒），1-190
        :param output_path: 保存路径，默认自动生成
        :param model: "stable-audio-2" 或 "stable-audio-2.5"
        :param output_format: "mp3" 或 "wav"
        :param seed: 随机种子，0表示随机
        :param steps: 采样步数（model=2时30-100，2.5时4-8）
        :param cfg_scale: prompt 遵循度（2时1-25，2.5时1-2）
        """
        if not self.available:
            return None

        duration = min(duration, self.max_duration)
        url = f"{self.base_url}/v2beta/audio/stable-audio-2/text-to-audio"

        fields = {
            "prompt": prompt,
            "duration": str(duration),
            "model": model,
            "output_format": output_format,
        }
        if seed != 0:
            fields["seed"] = str(seed)
        if steps is not None:
            fields["steps"] = str(steps)
        if cfg_scale is not None:
            fields["cfg_scale"] = str(cfg_scale)

        encoder = MultipartEncoder(fields=fields)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "audio/*",
            "Content-Type": encoder.content_type,
        }

        def _sync_post():
            response = requests.post(url, headers=headers, data=encoder, timeout=120)
            if response.status_code != 200:
                logger.error(f"StableAudio API error {response.status_code}: {response.text}")
                return None
            return response.content

        audio_bytes = await asyncio.to_thread(_sync_post)
        if audio_bytes is None:
            return None

        if output_path is None:
            import time
            output_path = Path(config.paths.output_dir) / f"stable_{int(time.time())}.{output_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"Saved to {output_path}")
        return output_path

    async def audio_to_audio(
        self,
        input_audio_path: Path,
        prompt: str,
        strength: float = 0.7,
        output_path: Optional[Path] = None,
        duration: Optional[int] = None,
        model: str = "stable-audio-2.5",
        output_format: str = "wav",
        seed: int = 0,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
    ) -> Optional[Path]:
        """
        基于现有音频进行转换
        :param input_audio_path: 输入音频文件（mp3/wav）
        :param prompt: 期望的输出音频描述
        :param strength: 强度（0-1），1完全重生成，0完全保留输入
        :param output_path: 输出路径
        :param duration: 输出时长（秒），默认与输入相同
        :param model: "stable-audio-2" 或 "stable-audio-2.5"
        :param output_format: "mp3" 或 "wav"
        :param seed: 随机种子
        :param steps: 采样步数
        :param cfg_scale: 遵循度
        """
        if not self.available:
            return None
        if not input_audio_path.exists():
            logger.error(f"Input audio file not found: {input_audio_path}")
            return None

        url = f"{self.base_url}/v2beta/audio/stable-audio-2/audio-to-audio"

        fields = {
            "prompt": prompt,
            "strength": str(strength),
            "model": model,
            "output_format": output_format,
        }
        if duration is not None:
            fields["duration"] = str(min(duration, self.max_duration))
        if seed != 0:
            fields["seed"] = str(seed)
        if steps is not None:
            fields["steps"] = str(steps)
        if cfg_scale is not None:
            fields["cfg_scale"] = str(cfg_scale)

        # 注意：文件字段需要特殊处理
        # 使用 MultipartEncoder 时，文件需要以 (filename, fileobj, content_type) 形式添加
        # 但 MultipartEncoder 不支持动态添加文件，所以改用 requests 直接发送 multipart
        # 为了简化，使用 requests 的 files 参数，并手动设置 Accept 头
        # 但为了保持异步，仍然放到线程池
        def _sync_upload():
            with open(input_audio_path, "rb") as f:
                files = {"audio": (input_audio_path.name, f, "audio/mpeg" if input_audio_path.suffix == ".mp3" else "audio/wav")}
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "audio/*",
                }
                response = requests.post(url, headers=headers, files=files, data=fields, timeout=180)
                if response.status_code != 200:
                    logger.error(f"StableAudio audio-to-audio error {response.status_code}: {response.text}")
                    return None
                return response.content

        audio_bytes = await asyncio.to_thread(_sync_upload)
        if audio_bytes is None:
            return None

        if output_path is None:
            import time
            output_path = Path(config.paths.output_dir) / f"stable_remix_{int(time.time())}.{output_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"Saved to {output_path}")
        return output_path