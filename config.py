"""Configuration Module for Multi-Agent Music System."""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIConfig:
    # Suno API 配置（非官方 API，已失效，保留作为兼容）
    suno_api_url: Optional[str] = os.getenv("SUNO_API_URL")
    suno_api_key: Optional[str] = os.getenv("SUNO_API_KEY")

    # Mureka API 配置（国产替代，优先使用）
    mureka_api_url: Optional[str] = os.getenv("MUREKA_API_URL")
    mureka_api_key: Optional[str] = os.getenv("MUREKA_API_KEY")

    # Udio API 配置（非官方）
    udio_api_url: Optional[str] = os.getenv("UDIO_API_URL")
    udio_api_key: Optional[str] = os.getenv("UDIO_API_KEY")
    udio_auth_token: Optional[str] = os.getenv("UDIO_AUTH_TOKEN")

    # 官方服务
    stability_api_key: Optional[str] = os.getenv("STABILITY_API_KEY")
    mubert_api_key: Optional[str] = os.getenv("MUBERT_API_KEY")
    mubert_api_url: str = os.getenv("MUBERT_API_URL", "https://api.mubert.com")
    replicate_api_token: Optional[str] = os.getenv("REPLICATE_API_TOKEN")

    # Google Vertex AI (Lyria 3)
    google_credentials_path: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    vertex_project_id: Optional[str] = os.getenv("VERTEX_AI_PROJECT_ID")
    vertex_location: str = os.getenv("VERTEX_AI_LOCATION", "us-central1")

    # OpenAI (用于创意简报)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # 智谱 AI (GLM) 歌词生成
    zhipuai_api_key: Optional[str] = os.getenv("ZHIPUAI_API_KEY")

    # 阿里云视频理解（Lyria 替代）
    aliyun_access_key: Optional[str] = os.getenv("ALIYUN_ACCESS_KEY")
    aliyun_access_secret: Optional[str] = os.getenv("ALIYUN_ACCESS_SECRET")
    aliyun_region: str = os.getenv("ALIYUN_REGION", "cn-shanghai")

    # 腾讯云 API 凭证（用于 MPS 音乐识别）
    tencent_secret_id: Optional[str] = os.getenv("TENCENT_SECRET_ID")
    tencent_secret_key: Optional[str] = os.getenv("TENCENT_SECRET_KEY")
    tencent_region: str = os.getenv("TENCENT_REGION", "ap-guangzhou")

    # 腾讯云智能作曲 API 端点（预留）
    tencent_music_api_endpoint: Optional[str] = os.getenv("TENCENT_MUSIC_API_ENDPOINT")

    # MiniMax Music API
    minimax_api_key: Optional[str] = os.getenv("MINIMAX_API_KEY")


@dataclass
class PathConfig:
    output_dir: str = "./output"
    temp_dir: str = "./temp"
    stems_dir: str = "./temp/stems"


@dataclass
class CostConfig:
    suno_per_gen: float = 0.00
    udio_per_gen: float = 0.00
    stable_audio_per_credit: float = 0.01
    mubert_per_stream: float = 0.00
    musicgen_per_sec: float = 0.001
    lyria_per_gen: float = 0.00
    stem_separation_per_file: float = 0.00
    audio_mixing_per_file: float = 0.00


@dataclass
class MixingConfig:
    vocal_gain: float = 3.0
    drum_gain: float = -3.0
    bass_gain: float = -2.0
    other_gain: float = -6.0


@dataclass
class SystemConfig:
    api: APIConfig = field(default_factory=APIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    mixing: MixingConfig = field(default_factory=MixingConfig)
    enable_fallback: bool = True
    fallback_to_pure_instrumental: bool = True
    keep_temp_files: bool = False
    sample_rate: int = 44100
    bitrate: str = "320k"
    # 输出格式配置
    output_format: str = "wav"  # "wav" 或 "mp3"
    output_bitrate: str = "320k"  # MP3 比特率


config = SystemConfig()