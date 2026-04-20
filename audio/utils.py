"""Audio utility functions."""

import subprocess
import json
from pathlib import Path
from typing import Optional

from pydub import AudioSegment


def get_audio_duration(audio_path: Path) -> float:
    """Get duration in seconds."""
    try:
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0
    except Exception:
        return 0.0


def convert_to_wav(input_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
    """Convert any audio to WAV."""
    if output_path is None:
        output_path = input_path.with_suffix('.wav')
    try:
        audio = AudioSegment.from_file(str(input_path))
        audio.export(str(output_path), format="wav")
        return output_path
    except Exception as e:
        print(f"Conversion failed: {e}")
        return None


def get_audio_info(audio_path: Path) -> dict:
    """Get metadata using ffprobe."""
    info = {"duration": 0.0, "sample_rate": 0, "channels": 0, "bitrate": 0}
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(audio_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    info["duration"] = float(stream.get("duration", 0))
                    info["sample_rate"] = int(stream.get("sample_rate", 0))
                    info["channels"] = int(stream.get("channels", 0))
                    info["bitrate"] = int(stream.get("bit_rate", 0)) // 1000
                    break
    except Exception:
        pass
    return info