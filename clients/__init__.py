# clients/__init__.py
from clients.udio_client import UdioClient
from clients.stable_audio_client import StableAudioClient
from clients.mubert_client import MubertClient
from clients.musicgen_client import MusicGenClient
from clients.lyria_client import LyriaClient

# 安全导入 SunoClient，避免因文件缺失或类不存在而崩溃
try:
    from clients.suno_client import SunoClient
except (ImportError, AttributeError):
    # 如果导入失败，定义一个虚拟的 SunoClient 类
    class SunoClient:
        def __init__(self, *args, **kwargs):
            pass
        async def generate(self, *args, **kwargs):
            return None

__all__ = [
    "SunoClient",
    "UdioClient",
    "StableAudioClient",
    "MubertClient",
    "MusicGenClient",
    "LyriaClient",
]