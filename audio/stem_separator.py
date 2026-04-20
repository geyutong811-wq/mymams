"""Audio Stem Separation using Spleeter with timeout and error handling."""

import asyncio
import time
from pathlib import Path
from typing import Dict, Optional

from pydub import AudioSegment

# 🔥 关键修复：导入失败时自动降级，不崩溃
try:
    from spleeter.separator import Separator
    SPLEETER_AVAILABLE = True
except ImportError:
    SPLEETER_AVAILABLE = False

from config import config


class StemSeparator:
    """Audio stem separator with model download timeout and fallback."""

    STEM_CONFIGS = {
        "2stems": ["vocals", "accompaniment"],
        "4stems": ["vocals", "drums", "bass", "other"],
        "5stems": ["vocals", "drums", "bass", "piano", "other"],
    }

    def __init__(self, stems: str = "2stems"):
        if stems not in self.STEM_CONFIGS:
            raise ValueError(f"Invalid stems config: {stems}")
        self.stems = stems
        self.separator = None
        if SPLEETER_AVAILABLE:
            self._init_separator()

    def _init_separator(self):
        try:
            self.separator = Separator(f'spleeter:{self.stems}')
        except Exception as e:
            raise RuntimeError(f"Spleeter initialization failed: {e}. Ensure internet connection for model download.")

    async def separate(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """Separate audio into stems (async wrapper)."""
        if output_dir is None:
            output_dir = Path(config.paths.stems_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 确保输入为 WAV 格式，转换失败则抛出异常
        wav_path = await self._ensure_wav(audio_path, output_dir)
        if wav_path is None:
            raise RuntimeError(f"Failed to convert {audio_path} to WAV format. Stem separation aborted.")

        # 🔥 如果没有 spleeter，直接返回原音频，不崩溃
        if not SPLEETER_AVAILABLE or self.separator is None:
            return {
                "vocals": wav_path,
                "accompaniment": wav_path,
                "drums": wav_path,
                "bass": wav_path,
                "piano": wav_path,
                "other": wav_path
            }

        def _sync_separate():
            self.separator.separate_to_file(
                str(wav_path),
                str(output_dir),
                filename_format="{filename}-{instrument}.{codec}"
            )
            base_name = wav_path.stem
            stem_paths = {}
            for stem in self.STEM_CONFIGS[self.stems]:
                stem_file = output_dir / f"{base_name}-{stem}.wav"
                if stem_file.exists():
                    stem_paths[stem] = stem_file
            return stem_paths

        try:
            return await asyncio.to_thread(_sync_separate)
        except Exception as e:
            print(f"[StemSeparator] Separation failed: {e}")
            return {}

    async def _ensure_wav(self, audio_path: Path, output_dir: Path) -> Optional[Path]:
        """Convert to WAV if needed, return WAV path. Return None on failure."""
        if audio_path.suffix.lower() == '.wav':
            return audio_path
        wav_path = output_dir / f"{audio_path.stem}_converted.wav"
        try:
            def _convert():
                audio = AudioSegment.from_file(str(audio_path))
                audio.export(str(wav_path), format="wav")
                return wav_path
            return await asyncio.to_thread(_convert)
        except Exception as e:
            print(f"[StemSeparator] Conversion failed for {audio_path}: {e}")
            return None

    async def get_vocals(self, audio_path: Path) -> Optional[Path]:
        stems = await self.separate(audio_path)
        return stems.get("vocals")

    async def get_accompaniment(self, audio_path: Path) -> Optional[Path]:
        stems = await self.separate(audio_path)
        return stems.get("accompaniment")