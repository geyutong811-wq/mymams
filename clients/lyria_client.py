"""腾讯云 MPS 音乐识别客户端 - 基于能跑通的测试脚本重构"""

import asyncio
import aiohttp
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

from config import config
from utils.logger import setup_logger

logger = setup_logger("LyriaClient")


class LyriaClient:
    def __init__(self):
        self.secret_id = config.api.tencent_secret_id
        self.secret_key = config.api.tencent_secret_key
        self.region = getattr(config.api, 'tencent_region', 'ap-guangzhou')
        self.endpoint = "mps.tencentcloudapi.com"
        self.service = "mps"
        self.version = "2019-06-12"

        if not self.secret_id or not self.secret_key:
            logger.warning("腾讯云凭证未配置，Lyria 客户端将无法工作。")

    def _sign(self, key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    def _get_headers(self, action: str, payload: str) -> dict:
        timestamp = int(time.time())
        date = time.strftime("%Y-%m-%d", time.gmtime(timestamp))
        ct = "application/json; charset=utf-8"
        canonical_headers = f"content-type:{ct}\nhost:{self.endpoint}\n"
        signed_headers = "content-type;host"
        hashed_payload = hashlib.sha256(payload.encode()).hexdigest()
        canonical_request = f"POST\n/\n\n{canonical_headers}\n{signed_headers}\n{hashed_payload}"
        algorithm = "TC3-HMAC-SHA256"
        credential_scope = f"{date}/{self.service}/tc3_request"
        hashed_canon = hashlib.sha256(canonical_request.encode()).hexdigest()
        string_to_sign = f"{algorithm}\n{timestamp}\n{credential_scope}\n{hashed_canon}"

        k_date = self._sign(("TC3" + self.secret_key).encode(), date)
        k_service = self._sign(k_date, self.service)
        k_signing = self._sign(k_service, "tc3_request")
        signature = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()

        authorization = (f"{algorithm} Credential={self.secret_id}/{credential_scope}, "
                         f"SignedHeaders={signed_headers}, Signature={signature}")
        return {
            "Authorization": authorization,
            "Content-Type": ct,
            "Host": self.endpoint,
            "X-TC-Action": action,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": self.version,
            "X-TC-Region": self.region,
        }

    async def _request(self, action: str, payload: dict) -> dict:
        payload_str = json.dumps(payload)
        headers = self._get_headers(action, payload_str)
        url = f"https://{self.endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload_str) as resp:
                text = await resp.text()
                if resp.status != 200:
                    logger.error(f"API 请求失败 (HTTP {resp.status}): {text}")
                    return {}
                try:
                    return json.loads(text)
                except:
                    return {"raw": text}

    async def analyze_video(self, video_url: str) -> Dict[str, Any]:
        """
        分析音频/视频文件，识别其中的歌曲信息。
        返回格式与 Lyria 原接口兼容。
        """
        if not self.secret_id or not self.secret_key:
            return {
                "analyzed": False,
                "tempo": 120,
                "key": "C major",
                "inferred_mood": "neutral",
                "tags": [],
                "note": "腾讯云凭证未配置"
            }

        logger.info(f"开始音乐识别: {video_url}")

        # 必须提供 OutputStorage（即使是占位）
        # 注意：Bucket 名称必须符合 COS 命名规则（如 dummy-123456789），Region 必须与你的 MPS 服务区域一致
        # 这里使用一个占位 bucket（不会实际写入数据）
        dummy_bucket = f"dummy-{self.region.replace('-', '')}-123456789"
        payload = {
            "InputInfo": {
                "Type": "URL",
                "UrlInputInfo": {"Url": video_url}
            },
            "OutputStorage": {
                "Type": "COS",
                "CosOutputStorage": {
                    "Bucket": dummy_bucket,
                    "Region": self.region
                }
            },
            "OutputDir": "/",
            "AiRecognitionTask": {
                "Definition": 10   # 音乐识别预设模板 ID
            }
        }

        # 提交任务
        result = await self._request("ProcessMedia", payload)
        if "Response" in result and "Error" in result["Response"]:
            err = result["Response"]["Error"]
            logger.error(f"API 错误: {err.get('Code')} - {err.get('Message')}")
            return {"analyzed": False, "error": err.get("Message")}

        task_id = result.get("Response", {}).get("TaskId")
        if not task_id:
            logger.error(f"未获取到 TaskId: {result}")
            return {"analyzed": False, "error": "未获取到任务ID"}

        logger.info(f"任务已提交，TaskId: {task_id}，等待结果...")
        analysis_result = await self._poll_result(task_id)
        return analysis_result

    async def _poll_result(self, task_id: str, max_wait: int = 60) -> Dict[str, Any]:
        """轮询 DescribeTaskDetail 获取识别结果"""
        start = time.time()
        delay = 2.0
        while time.time() - start < max_wait:
            payload = {"TaskId": task_id}
            result = await self._request("DescribeTaskDetail", payload)
            if not result:
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 10)
                continue

            resp = result.get("Response", {})
            status = resp.get("Status")
            if status == "FINISH":
                # 提取音乐识别结果
                ai_recog = resp.get("AiRecognitionResultSet", [])
                for item in ai_recog:
                    if item.get("Type") == "MusicRecognition":
                        music_info = item.get("MusicRecognitionResult", {})
                        # 可能包含多个识别出的歌曲，取第一个
                        items = music_info.get("Items", [])
                        if items:
                            best = items[0]
                            song_name = best.get("Name", "")
                            singer = best.get("Singer", "")
                            album = best.get("Album", "")
                            # 根据歌名简单推断情绪
                            mood = self._infer_mood_from_song(song_name)
                            return {
                                "analyzed": True,
                                "song_name": song_name,
                                "singer": singer,
                                "album": album,
                                "inferred_mood": mood,
                                "tempo": 120,
                                "key": "C major",
                                "tags": [song_name],
                                "raw": result
                            }
                # 没有匹配到歌曲
                return {"analyzed": True, "matched": False, "message": "未识别到歌曲", "raw": result}
            elif status == "FAIL":
                logger.error(f"任务失败: {resp.get('Message')}")
                return {"analyzed": False, "error": resp.get("Message")}
            else:
                logger.info(f"任务处理中，状态: {status}")
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 10)

        logger.error("轮询超时")
        return {"analyzed": False, "error": "轮询超时"}

    def _infer_mood_from_song(self, song_name: str) -> str:
        if not song_name:
            return "neutral"
        name = song_name.lower()
        if any(w in name for w in ["happy", "joy", "smile", "sun", "阳光", "快乐"]):
            return "uplifting"
        if any(w in name for w in ["sad", "cry", "tear", "broken", "悲伤", "泪"]):
            return "melancholic"
        if any(w in name for w in ["love", "heart", "romantic", "爱", "心"]):
            return "romantic"
        if any(w in name for w in ["energy", "power", "rock", "激情", "动感"]):
            return "energetic"
        if any(w in name for w in ["calm", "peace", "dream", "安静", "宁静"]):
            return "relaxing"
        return "neutral"

    # 兼容原有接口，音乐识别不支持生成
    async def generate(self, prompt: str, duration: int = 30, output_path: Optional[Path] = None) -> Optional[Path]:
        logger.warning("音乐识别客户端不支持生成音频，请使用其他客户端生成音乐。")
        return None