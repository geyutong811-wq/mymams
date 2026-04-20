"""Audio mixing utilities with alignment and LUFS normalization."""

import asyncio
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False
    print("Warning: pyloudnorm not installed. Using peak normalization instead.")

from config import config


class AudioMixer:
    """Audio mixer with format alignment and output format conversion."""

    def __init__(self):
        self.temp_dir = Path(config.paths.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _align_tracks(self, *tracks: AudioSegment) -> List[AudioSegment]:
        if not tracks:
            return []
        target_rate = max(t.frame_rate for t in tracks)
        target_channels = max(t.channels for t in tracks)
        aligned = []
        for t in tracks:
            if t.frame_rate != target_rate:
                t = t.set_frame_rate(target_rate)
            if t.channels != target_channels:
                t = t.set_channels(target_channels)
            aligned.append(t)
        max_len = max(len(t) for t in aligned)
        for i, t in enumerate(aligned):
            if len(t) < max_len:
                aligned[i] = t + AudioSegment.silent(duration=max_len - len(t))
        return aligned

    async def overlay_audio(
        self,
        main_audio_path: Path,
        overlay_audio_path: Path,
        position_ms: int = 0,
        volume_adjustment_db: int = 0,
        output_path: Optional[Path] = None
    ) -> Path:
        def _load_and_mix():
            main = AudioSegment.from_file(str(main_audio_path))
            overlay = AudioSegment.from_file(str(overlay_audio_path))
            main, overlay = self._align_tracks(main, overlay)
            if volume_adjustment_db != 0:
                overlay = overlay + volume_adjustment_db
            mixed = main.overlay(overlay, position=position_ms)
            if output_path is None:
                out = self.temp_dir / f"overlay_{int(time.time())}.wav"
            else:
                out = output_path
            out.parent.mkdir(parents=True, exist_ok=True)
            mixed.export(str(out), format="wav")
            return out
        return await asyncio.to_thread(_load_and_mix)

    async def mix_stems(
        self,
        stem_paths: List[Tuple[Path, int]],
        output_path: Optional[Path] = None
    ) -> Path:
        def _mix():
            segments = [AudioSegment.from_file(str(p)) for p, _ in stem_paths]
            volumes = [vol for _, vol in stem_paths]
            aligned = self._align_tracks(*segments)
            mixed = aligned[0] + volumes[0] if volumes[0] != 0 else aligned[0]
            for i in range(1, len(aligned)):
                seg = aligned[i] + volumes[i] if volumes[i] != 0 else aligned[i]
                mixed = mixed.overlay(seg)
            if output_path is None:
                out = self.temp_dir / f"mix_{int(time.time())}.wav"
            else:
                out = output_path
            out.parent.mkdir(parents=True, exist_ok=True)
            mixed.export(str(out), format="wav")
            return out
        return await asyncio.to_thread(_mix)

    async def normalize_loudness(
        self,
        audio_path: Path,
        target_lufs: float = -14.0,
        output_path: Optional[Path] = None
    ) -> Path:
        def _normalize():
            audio = AudioSegment.from_file(str(audio_path))
            use_pyloudnorm = HAS_PYLOUDNORM and audio.channels in [1, 2]
            if not use_pyloudnorm and HAS_PYLOUDNORM:
                print(f"[AudioMixer] Warning: LUFS normalization skipped for {audio.channels}-channel audio. Falling back to peak normalization.")
            if use_pyloudnorm:
                samples = np.array(audio.get_array_of_samples())
                if audio.channels == 2:
                    samples = samples.reshape(-1, 2)
                sample_rate = audio.frame_rate
                if samples.dtype == np.int16:
                    samples_float = samples.astype(np.float64) / 32768.0
                else:
                    samples_float = samples.astype(np.float64)
                meter = pyln.Meter(sample_rate)
                loudness = meter.integrated_loudness(samples_float)
                if np.isfinite(loudness):
                    gain = target_lufs - loudness
                    gain_linear = 10 ** (gain / 20.0)
                    samples_float = samples_float * gain_linear
                    samples_float = np.clip(samples_float, -1.0, 1.0)
                    if audio.channels == 2:
                        samples_int = (samples_float * 32767).astype(np.int16).flatten()
                    else:
                        samples_int = (samples_float * 32767).astype(np.int16)
                    normalized = AudioSegment(
                        samples_int.tobytes(),
                        frame_rate=sample_rate,
                        sample_width=2,
                        channels=audio.channels
                    )
                else:
                    normalized = pydub_normalize(audio)
            else:
                normalized = pydub_normalize(audio)
            if output_path is None:
                out = self.temp_dir / f"normalized_{int(time.time())}.wav"
            else:
                out = output_path
            out.parent.mkdir(parents=True, exist_ok=True)
            normalized.export(str(out), format="wav")
            return out
        return await asyncio.to_thread(_normalize)

    async def apply_fade(
        self,
        audio_path: Path,
        fade_in_ms: int = 0,
        fade_out_ms: int = 0,
        output_path: Optional[Path] = None
    ) -> Path:
        def _fade():
            audio = AudioSegment.from_file(str(audio_path))
            if fade_in_ms > 0:
                audio = audio.fade_in(fade_in_ms)
            if fade_out_ms > 0:
                audio = audio.fade_out(fade_out_ms)
            if output_path is None:
                out = self.temp_dir / f"fade_{int(time.time())}.wav"
            else:
                out = output_path
            out.parent.mkdir(parents=True, exist_ok=True)
            audio.export(str(out), format="wav")
            return out
        return await asyncio.to_thread(_fade)

    async def save_as_format(
        self,
        audio_path: Path,
        output_format: str = None,
        bitrate: str = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Convert audio to specified format (wav or mp3).
        If output_format is None, uses config.output_format.
        If output_path is None, saves to config.paths.output_dir with a unique name.
        """
        fmt = (output_format or config.output_format).lower()
        br = bitrate or config.output_bitrate
        if fmt not in ["wav", "mp3"]:
            fmt = "wav"

        def _convert():
            audio = AudioSegment.from_file(str(audio_path))
            if output_path is None:
                # 默认保存到 output_dir，而不是 temp_dir
                out = Path(config.paths.output_dir) / f"{audio_path.stem}_final.{fmt}"
            else:
                out = output_path
            out.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "mp3":
                audio.export(str(out), format="mp3", bitrate=br)
            else:
                audio.export(str(out), format="wav")
            return out
        return await asyncio.to_thread(_convert)