"""Mubert API Client - 调用 MiniMax Music API（music-2.5+ 纯音乐模式）"""

import asyncio
import requests
import time
from pathlib import Path
from typing import Optional

from config import config
from utils.logger import setup_logger

logger = setup_logger("MubertClient")


class MubertClient:
    def __init__(self):
        self.api_key = config.api.minimax_api_key
        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY not set. Please add to .env file.")
        self.base_url = "https://api.minimaxi.com"
        # 如需代理，在这里设置（例如 "http://127.0.0.1:7890"）
        self.proxy = None

    async def generate_stream(
        self,
        genre: str = "electronic",
        mood: str = "uplifting",
        duration: int = 60,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        prompt = f"{genre}, {mood}, background music, instrumental"

        url = f"{self.base_url}/v1/music_generation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",  # 关键：禁用压缩，避免 brotli 解码错误
        }
        payload = {
            "model": "music-2.5+",
            "prompt": prompt,
            "is_instrumental": True,
            "output_format": "hex",
            "audio_setting": {
                "sample_rate": 44100,
                "bitrate": 256000,
                "format": "mp3",
            },
        }

        max_retries = 3
        timeout = 180  # 增加超时到 180 秒

        for attempt in range(max_retries):
            def _sync_post():
                try:
                    proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                        proxies=proxies
                    )
                    return response
                except Exception as e:
                    logger.error(f"Request attempt {attempt+1} failed: {e}")
                    return None

            response = await asyncio.to_thread(_sync_post)
            if response is None:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.info(f"Retrying in {wait} seconds...")
                    await asyncio.sleep(wait)
                continue

            if response.status_code == 200:
                break
            else:
                # 非 200 错误，不再重试（权限、余额等问题）
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
        else:
            logger.error("All retries failed.")
            return None

        try:
            result = response.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None

        base_resp = result.get("base_resp", {})
        if base_resp.get("status_code") != 0:
            error_msg = base_resp.get("status_msg", "Unknown error")
            logger.error(f"MiniMax API returned error: {error_msg}")
            return None

        audio_hex = result.get("data", {}).get("audio")
        if not audio_hex:
            logger.error("No audio data in response")
            return None

        try:
            audio_bytes = bytes.fromhex(audio_hex)
        except ValueError as e:
            logger.error(f"Failed to decode hex audio data: {e}")
            return None

        if output_path is None:
            output_path = Path(config.paths.output_dir) / f"minimax_{int(time.time())}.mp3"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        logger.info(f"Background music saved to {output_path}")
        return output_path