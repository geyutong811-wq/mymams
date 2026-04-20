# test_lyrics.py
import asyncio
from utils.lyrics_generator import LyricsGenerator

async def test():
    gen = LyricsGenerator()
    lyrics = await gen.generate_lyrics(
        prompt="A peaceful piano melody",
        genre="ambient",
        mood="calm",
        duration=30,
        is_instrumental=True
    )
    print("Generated lyrics:\n", lyrics)

asyncio.run(test())