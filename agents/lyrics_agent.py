"""Lyrics Agent - 生成与音乐主题相关的歌词"""

from typing import Any, Dict, Optional
from agents.base_agent import BaseAgent
from utils.lyrics_generator import LyricsGenerator


class LyricsAgent(BaseAgent):
    def __init__(self):
        super().__init__("LyricsAgent")
        self.generator = LyricsGenerator()

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = input_data.get("prompt", "")
        genre = input_data.get("genre", "pop")
        mood = input_data.get("mood", "neutral")
        duration = input_data.get("duration", 60)

        self.log(f"Generating lyrics for: {prompt[:50]}...")
        lyrics = await self.generator.generate_lyrics(
            prompt=prompt,
            genre=genre,
            mood=mood,
            duration=duration,
            is_instrumental=False  # 有歌词
        )
        return {
            "lyrics": lyrics,
            "prompt": prompt,
            "genre": genre,
            "mood": mood,
            "duration": duration
        }