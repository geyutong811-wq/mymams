"""Accompaniment Agent - 生成纯音乐伴奏"""

from typing import Any, Dict
from agents.base_agent import BaseAgent
from clients.mubert_client import MubertClient  # 修正：使用 mubert_client 而非 minimax_client


class AccompanimentAgent(BaseAgent):
    def __init__(self):
        super().__init__("AccompanimentAgent")
        self.client = MubertClient()

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        genre = input_data.get("genre", "pop")
        mood = input_data.get("mood", "neutral")
        duration = input_data.get("duration", 60)
        lyrics = input_data.get("lyrics", "")  # 可选
        # ==========================================
        # 【修改点：开始】
        # 如果情绪是忧郁 (melancholic)，强制在传给底层客户端的参数中加入无鼓点限制
        if mood == "melancholic":
            mood = f"{mood}, drumless, no percussion, ambient pad"
        # 【修改点：结束】
        # ==========================================
        self.log(f"Generating accompaniment for {genre} {mood}...")
        audio_path = await self.client.generate_stream(
            genre=genre,
            mood=mood,
            duration=duration
        )
        return {"accompaniment_audio": audio_path}