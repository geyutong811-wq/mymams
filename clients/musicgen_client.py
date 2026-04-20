"""MusicGen Client - 改用 ElevenLabs Music API (via Replicate)"""

import asyncio
import aiohttp
from pathlib import Path
from typing import Optional

import replicate

from config import config
from utils.logger import setup_logger

logger = setup_logger("MusicGenClient")


class MusicGenClient:
    """ElevenLabs Music API 客户端（通过 Replicate）"""

    MODEL_ID = "elevenlabs/music"

    def __init__(self):
        self.api_token = config.api.replicate_api_token
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN missing")
        replicate.api_token = self.api_token

    async def generate(
        self,
        prompt: str,
        duration: int = 30,
        output_path: Optional[Path] = None,
        output_format: str = "wav_cd_quality",   # 可选: mp3, wav_cd_quality
        instrumental: bool = False,              # 纯音乐模式
    ) -> Optional[Path]:
        """
        使用 ElevenLabs Music API 生成音乐
        :param prompt: 音乐描述文本
        :param duration: 时长（秒），最多 60 秒（根据官方限制）
        :param output_path: 保存路径
        :param output_format: 输出格式（wav_cd_quality, mp3）
        :param instrumental: 是否生成纯音乐（True=无歌词，False=可能有歌词）
        """
        # 限制时长（官方最大 60 秒，可根据实际情况调整）
        max_duration = 60
        if duration > max_duration:
            logger.warning(f"Requested {duration}s, but ElevenLabs Music supports max {max_duration}s. Truncating.")
            duration = max_duration

        input_params = {
            "prompt": prompt,
            "music_length_ms": duration * 1000,
            "output_format": output_format,
            "force_instrumental": instrumental,
        }

        def _sync_run():
            return replicate.run(self.MODEL_ID, input=input_params)

        try:
            output = await asyncio.to_thread(_sync_run)
            # replicate 返回的 output 对象有 url 属性
            audio_url = output.url() if hasattr(output, "url") else output.url
            if not audio_url:
                logger.error("No audio URL returned from ElevenLabs API")
                return None

            if output_path is None:
                output_path = Path(config.paths.output_dir) / f"musicgen_{int(asyncio.get_event_loop().time())}.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiohttp.ClientSession() as session:
                async with session.get(audio_url) as resp:
                    if resp.status == 200:
                        with open(output_path, "wb") as f:
                            f.write(await resp.read())
                        logger.info(f"Saved to {output_path}")
                        return output_path
                    else:
                        logger.error(f"Failed to download audio: HTTP {resp.status}")
                        return None

        except Exception as e:
            logger.error(f"ElevenLabs Music generation failed: {e}")
            return None

    async def remix(self, audio_path: Path, prompt: str, mix_weight: float = 0.7) -> Optional[Path]:
        """
        ElevenLabs Music API 不支持 remix（音频到音频编辑），此方法仅作占位。
        如需 remix，请使用 Stable Audio 的 audio-to-audio 功能。
        """
        logger.warning("ElevenLabs Music API does not support remix. Use StableAudioClient for audio-to-audio.")
        return None