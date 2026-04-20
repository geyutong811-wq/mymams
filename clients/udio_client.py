"""Udio API Client - 优化版，解决轮询超时问题"""

import asyncio
import aiohttp
from pathlib import Path
from typing import Optional

from config import config
from utils.logger import setup_logger

logger = setup_logger("UdioClient")


class UdioClient:
    def __init__(self):
        self.api_key = config.api.udio_api_key
        self.base_url = config.api.udio_api_url
        if not self.api_key or not self.base_url:
            logger.warning("Udio 未配置，将使用静音占位符。")
            self.available = False
        else:
            self.available = True
        self.proxy = None  # 如需代理可设置

    async def generate(
        self,
        prompt: str,
        style: str = "pop",
        duration: int = 60,
        output_path: Optional[Path] = None,
        max_wait_seconds: int = 300,  # 增加到 5 分钟
    ) -> Optional[Path]:
        if not self.available:
            return await self._silent_fallback(duration, output_path)

        logger.info(f"生成音乐: '{prompt[:50]}...'")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "gpt_description_prompt": prompt,
            "make_instrumental": style == "instrumental",
            "callback_url": "",
        }
        api_endpoint = f"{self.base_url}/api/generate"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(api_endpoint, headers=headers, json=payload, proxy=self.proxy) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"API 请求失败 (HTTP {resp.status}): {text}")
                        return await self._silent_fallback(duration, output_path)
                    result = await resp.json()

                task_id = result.get("workId")
                if not task_id:
                    logger.error("未找到任务 ID")
                    return await self._silent_fallback(duration, output_path)

                logger.info(f"任务已提交，ID: {task_id}，开始轮询（最长 {max_wait_seconds} 秒）...")
                audio_url = await self._poll_result(session, task_id, max_wait_seconds)
                if not audio_url:
                    logger.error("未能获取音频 URL")
                    return await self._silent_fallback(duration, output_path)

                if output_path is None:
                    output_path = Path(config.paths.output_dir) / f"udio_{int(asyncio.get_event_loop().time())}.mp3"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                async with session.get(audio_url, proxy=self.proxy) as audio_resp:
                    if audio_resp.status == 200:
                        with open(output_path, 'wb') as f:
                            f.write(await audio_resp.read())
                        logger.info(f"音乐已保存至: {output_path}")
                        return output_path
                    else:
                        logger.error(f"下载音频失败 (HTTP {audio_resp.status})")
                        return await self._silent_fallback(duration, output_path)

            except Exception as e:
                logger.error(f"生成过程异常: {e}")
                return await self._silent_fallback(duration, output_path)

    async def _poll_result(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
        max_wait_seconds: int = 300,
        initial_delay: float = 2.0,
        max_delay: float = 15.0,
    ) -> Optional[str]:
        """轮询 /api/v2/feed 接口，动态增加延迟"""
        status_url = f"{self.base_url}/api/v2/feed"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        start_time = asyncio.get_event_loop().time()
        delay = initial_delay

        while (asyncio.get_event_loop().time() - start_time) < max_wait_seconds:
            try:
                async with session.get(status_url, headers=headers, params={"workId": task_id}, proxy=self.proxy) as resp:
                    if resp.status != 200:
                        logger.warning(f"轮询请求失败 (HTTP {resp.status})，等待 {delay:.1f}s 重试...")
                        await asyncio.sleep(delay)
                        delay = min(delay * 1.5, max_delay)
                        continue

                    data = await resp.json()
                    inner_data = data.get("data", {})
                    task_type = inner_data.get("type")
                    response_data = inner_data.get("response_data")

                    if task_type == "SUCCESS" and response_data and len(response_data) > 0:
                        audio_url = response_data[0].get("audio_url")
                        if audio_url:
                            logger.info("任务完成，获取到音频 URL")
                            return audio_url
                        else:
                            logger.error("任务成功但未找到音频 URL")
                            return None
                    elif task_type == "FAILED":
                        error_msg = inner_data.get("error_message", "未知错误")
                        logger.error(f"任务失败: {error_msg}")
                        return None
                    else:
                        elapsed = asyncio.get_event_loop().time() - start_time
                        logger.info(f"任务状态: {task_type or 'IN_PROGRESS'}，已等待 {elapsed:.0f}s，下次重试 {delay:.1f}s 后")
                        await asyncio.sleep(delay)
                        delay = min(delay * 1.5, max_delay)

            except Exception as e:
                logger.warning(f"轮询异常: {e}，等待 {delay:.1f}s 重试...")
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, max_delay)

        logger.error(f"轮询超时（{max_wait_seconds} 秒）")
        return None

    async def _silent_fallback(self, duration: int, output_path: Optional[Path]) -> Path:
        """生成静音占位符"""
        from pydub import AudioSegment
        if output_path is None:
            output_path = Path(config.paths.output_dir) / f"udio_silent_{int(asyncio.get_event_loop().time())}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        silent = AudioSegment.silent(duration=duration * 1000)
        silent.export(str(output_path), format="wav")
        logger.info(f"生成静音占位符: {output_path}")
        return output_path