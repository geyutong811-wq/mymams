"""Research & Tuning Agent - Experimental music generation."""
from typing import Any, Dict, Optional
from pathlib import Path

from agents.base_agent import BaseAgent
from clients.musicgen_client import MusicGenClient


class ResearchAgent(BaseAgent):
    """Research & Tuning Agent for experimental music generation."""

    def __init__(self):
        super().__init__("Research")
        self.musicgen_client = MusicGenClient()

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate experimental music."""
        prompt = input_data.get("prompt", "")
        duration = input_data.get("duration", 30)
        remix_audio = input_data.get("remix_audio")
        mix_weight = input_data.get("mix_weight", 0.7)

        self.log(f"Research generation: {prompt[:50]}...")
        if remix_audio:
            self.log(f"Remixing {remix_audio} with weight {mix_weight}")
            audio_path = await self.musicgen_client.remix(
                audio_path=remix_audio,
                prompt=prompt,
                mix_weight=mix_weight
            )
        else:
            audio_path = await self.musicgen_client.generate(
                prompt=prompt,
                duration=duration
            )

        result = {
            "audio_path": audio_path,
            "metadata": {
                "prompt": prompt,
                "duration": duration,
                "model": "musicgen",
                "is_remix": remix_audio is not None,
            }
        }

        self.log(f"Research generation complete. Output: {audio_path}")
        return result