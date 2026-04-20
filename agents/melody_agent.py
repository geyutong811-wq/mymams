"""Melody Agent - 根据歌词生成主旋律/人声"""

from typing import Any, Dict, Optional
from pathlib import Path

from agents.base_agent import BaseAgent
from clients.udio_client import UdioClient  # 也可用 MiniMax


class MelodyAgent(BaseAgent):
    def __init__(self):
        super().__init__("MelodyAgent")
        self.client = UdioClient()  # 或使用其他音乐生成客户端

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        lyrics = input_data.get("lyrics", "")
        genre = input_data.get("genre", "pop")
        mood = input_data.get("mood", "neutral")
        duration = input_data.get("duration", 60)
        prompt = input_data.get("prompt", "")

        self.log(f"Generating melody with lyrics of length {len(lyrics)}...")
        # ==========================================
        # 【修改点：开始】
        # 拦截并修改发给 Udio 的 prompt。如果情绪忧郁或用户明确要求无鼓点，则加入限制词
        if mood == "melancholic" or "drumless" in prompt.lower():
            full_prompt = f"{genre}, {mood}, drumless, no percussion, pure ambient pad music. Lyrics: {lyrics[:200]}"
        else:
            full_prompt = f"{genre}, {mood} music. Lyrics: {lyrics[:200]}"
        # 【修改点：结束】
        # ==========================================
        audio_path = await self.client.generate(
            prompt=full_prompt,
            style=genre,
            duration=duration
        )
        return {
            "main_audio": audio_path,
            "lyrics": lyrics,
            "genre": genre,
            "mood": mood
        }