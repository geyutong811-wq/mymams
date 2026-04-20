"""歌词生成器 - 基于智谱 AI GLM 模型生成与音乐主题相关的歌词"""

import asyncio
from typing import Optional

from config import config
from utils.logger import setup_logger

logger = setup_logger("LyricsGenerator")

# 尝试导入智谱 AI SDK
try:
    from zhipuai import ZhipuAI
    ZHIPU_AVAILABLE = True
except ImportError:
    ZHIPU_AVAILABLE = False
    logger.warning("zhipuai not installed. Lyrics generation will use fallback. Install with: pip install zhipuai")


class LyricsGenerator:
    def __init__(self):
        self.api_key = config.api.zhipuai_api_key
        self.model = "glm-4-flash"  # 使用 GLM-4-Flash，速度快且成本低；也可改为 "glm-4-plus"
        self.available = ZHIPU_AVAILABLE and bool(self.api_key)
        if self.available:
            self.client = ZhipuAI(api_key=self.api_key)
        else:
            logger.warning("Lyrics generator not available (ZhipuAI API key missing or package not installed).")

    async def generate_lyrics(
        self,
        prompt: str,
        genre: str = "pop",
        mood: str = "uplifting",
        duration: int = 60,
        is_instrumental: bool = False
    ) -> str:
        """
        根据音乐描述生成歌词。
        如果 is_instrumental 为 True，则生成简短的吟唱词或意境描述。
        """
        if not self.available:
            return self._fallback_lyrics(genre, mood, is_instrumental)

        system_prompt = """你是一位专业的歌词创作人。根据用户的描述生成歌词。
歌词需符合给定的风格和情绪，保持简洁（约8-16行）。
如果用户要求生成纯音乐背景，则生成简短的哼唱词（如"啦 啦 啦"、"哦 哦 哦"）或对音乐意境的诗意描述，不要生成完整句子。"""

        user_prompt = f"""请根据以下要求生成歌词：
- 描述: {prompt}
- 风格: {genre}
- 情绪: {mood}
- 时长: {duration} 秒
- 是否为纯音乐: {is_instrumental}

{'请生成简短的哼唱词（如"啦 啦 啦"、"哦 哦 哦"）或简短的意境描述。' if is_instrumental else '请生成有意义的歌词。'}"""

        try:
            # 智谱 AI 的异步调用需要在线程池中运行，因为 SDK 是同步的
            def _sync_generate():
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.8,
                    max_tokens=300
                )
                return response.choices[0].message.content.strip()

            lyrics = await asyncio.to_thread(_sync_generate)
            return lyrics
        except Exception as e:
            logger.error(f"ZhipuAI lyrics generation failed: {e}")
            return self._fallback_lyrics(genre, mood, is_instrumental)

    def _fallback_lyrics(self, genre: str, mood: str, is_instrumental: bool) -> str:
        """降级方案：生成简单的占位歌词"""
        if is_instrumental:
            return f"[Instrumental] {genre} {mood} background music. La la la, oh oh oh."
        else:
            return f"[Verse]\nIn the {mood} {genre} sound\nPeace and rhythm all around\n\n[Chorus]\nFeel the music, let it flow\n{genre.capitalize()} vibes, take it slow"