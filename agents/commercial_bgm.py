"""Commercial BGM Agent - Generates background music with generated lyrics."""

from typing import Any, Dict, Optional
from pathlib import Path

from agents.base_agent import BaseAgent
from clients.mubert_client import MubertClient
from utils.lyrics_generator import LyricsGenerator


class CommercialBGMAgent(BaseAgent):
    def __init__(self):
        super().__init__("CommercialBGM")
        self.mubert_client = MubertClient()
        self.lyrics_gen = LyricsGenerator()

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        genre = input_data.get("genre", "electronic")
        mood = input_data.get("mood", "uplifting")
        duration = input_data.get("duration", 60)
        use_case = input_data.get("use_case", "general")
        prompt = input_data.get("prompt", f"{genre} {mood} background music")

        self.log(f"Generating background music: genre={genre}, mood={mood}, duration={duration}s")

        # 生成与音乐主题相关的歌词（纯音乐背景使用 vocalise）
        lyrics = await self.lyrics_gen.generate_lyrics(
            prompt=prompt,
            genre=genre,
            mood=mood,
            duration=duration,
            is_instrumental=True   # 背景音乐纯音乐，生成 vocalise
        )
        self.log(f"Generated lyrics: {lyrics[:100]}...")

        audio_path = await self.mubert_client.generate_stream(
            genre=genre,
            mood=mood,
            duration=duration,
            lyrics=lyrics
        )

        if audio_path is None:
            self.log("Failed to generate music, returning None")
            return {"audio_path": None, "metadata": {}}

        result = {
            "audio_path": audio_path,
            "metadata": {
                "genre": genre,
                "mood": mood,
                "duration": duration,
                "use_case": use_case,
                "model": "minimax-music-1.5",
                "royalty_free": True,
            }
        }
        self.log(f"Background music generated: {audio_path}")
        return result