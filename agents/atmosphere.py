"""Atmosphere & SFX Agent - Generates ambient sounds and sound effects."""

import time
from typing import Any, Dict, Optional
from pathlib import Path

from pydub import AudioSegment
from agents.base_agent import BaseAgent
from clients.stable_audio_client import StableAudioClient
from clients.mubert_client import MubertClient
from config import config


class AtmosphereAgent(BaseAgent):
    """Atmosphere & Sound Effects Agent with fallback to Mubert."""

    def __init__(self):
        super().__init__("Atmosphere")
        self.stable_audio_client = StableAudioClient()
        self.mubert_client = MubertClient()
        self.stable_available = self.stable_audio_client.available

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        duration = input_data.get("duration", 30)
        is_sfx = input_data.get("is_sfx", False)
        reference_audio = input_data.get("reference_audio")

        self.log(f"Generating atmosphere: {prompt[:50]}...")

        result = {}
        audio_path = None

        if not is_sfx:
            # 优先使用 Stable Audio
            if self.stable_available:
                audio_path = await self.stable_audio_client.generate(
                    prompt=prompt,
                    duration=min(duration, 30),
                    loop_to_duration=duration
                )
                if audio_path:
                    self.log("Generated with Stable Audio.")
            # 如果失败，回退到 Mubert
            if not audio_path:
                self.log("Stable Audio unavailable or failed, falling back to Mubert.")
                genre = "ambient"
                if "lo-fi" in prompt.lower() or "lofi" in prompt.lower():
                    genre = "lo-fi"
                elif "electronic" in prompt.lower():
                    genre = "electronic"
                elif "classical" in prompt.lower():
                    genre = "classical"
                audio_path = await self.mubert_client.generate_stream(
                    genre=genre,
                    mood="relaxing",
                    duration=duration
                )
                if audio_path:
                    result["fallback"] = True
                    self.log("Generated with Mubert as fallback.")

        if reference_audio and self.stable_available:
            transformed = await self.stable_audio_client.audio_to_audio(
                input_audio_path=reference_audio,
                prompt=prompt,
                strength=0.6 if is_sfx else 0.8
            )
            result["transformed_audio"] = transformed
            if not audio_path:
                audio_path = transformed

        # 最终兜底：静音占位符
        if audio_path is None:
            self.log("No audio generated. Creating silent placeholder.")
            silent_path = Path(config.paths.temp_dir) / f"silent_{int(time.time())}.wav"
            silent_path.parent.mkdir(parents=True, exist_ok=True)
            silent = AudioSegment.silent(duration=duration * 1000)
            silent.export(str(silent_path), format="wav")
            audio_path = silent_path
            result["is_placeholder"] = True

        result["audio_path"] = audio_path
        result["metadata"] = {
            "prompt": prompt,
            "duration": duration,
            "is_sfx": is_sfx,
            "model": "fallback" if result.get("fallback") else "stable-audio",
        }
        return result