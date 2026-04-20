#!/usr/bin/env python
"""Multi-Agent AI Music System - 主入口（增强版，带静音兜底）"""

import asyncio
import atexit
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

# 确保项目根目录在路径中
root_path = str(Path(__file__).parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from orchestration.supervisor import SupervisorAgent, WorkflowState
from config import config
from utils.logger import setup_logger

logger = setup_logger("Main", log_file=Path(config.paths.output_dir) / "main.log")


def serialize_workflow_result(state: WorkflowState) -> Dict[str, Any]:
    def safe_path(p) -> Any:
        if p is None:
            return None
        return str(Path(p).resolve())

    stems = None
    if state.stems:
        stems = {k: safe_path(v) for k, v in state.stems.items()}
    return {
        "success": state.error is None and state.final_audio is not None,
        "error": state.error,
        "total_cost": state.total_cost,
        "fallback_used": state.fallback_used,
        "final_audio_path": safe_path(state.final_audio),
        "creative_brief": state.creative_brief,
        "stems": stems,
        "current_step": state.current_step,
        "enable_stem_separation": state.enable_stem_separation,
        "enable_vocal_synthesis": state.enable_vocal_synthesis,
        "vocal_voice_type": state.vocal_voice_type,
    }


async def run_generation_async(user_input: Dict[str, Any]) -> WorkflowState:
    Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.temp_dir).mkdir(parents=True, exist_ok=True)
    supervisor = SupervisorAgent()
    return await supervisor.run(user_input)


def run_generation_sync(user_input: Dict[str, Any]) -> Dict[str, Any]:
    state = asyncio.run(run_generation_async(user_input))
    return serialize_workflow_result(state)


def cleanup_on_exit():
    if not config.keep_temp_files:
        for d in [config.paths.temp_dir, config.paths.stems_dir]:
            path = Path(d)
            if path.exists():
                try:
                    shutil.rmtree(path)
                    logger.info(f"Cleaned up {path} on exit")
                except Exception:
                    pass


async def main():
    print("=" * 60)
    print("Multi-Agent AI Music System")
    print("=" * 60)
    print()

    Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.temp_dir).mkdir(parents=True, exist_ok=True)

    atexit.register(cleanup_on_exit)

    print("Describe the music you want to generate:")
    prompt = input("> ").strip()
    if not prompt:
        logger.error("Empty prompt provided")
        print("Error: Please provide a description.")
        return

    print("\nOptional settings (press Enter to skip):")
    genre = input("Genre (pop/electronic/rock/jazz/ambient): ").strip() or None
    mood = input("Mood (uplifting/melancholic/energetic/relaxing): ").strip() or None

    duration_str = input("Duration in seconds (1-600, default 60): ").strip()
    if duration_str:
        try:
            duration = int(duration_str)
            if duration < 1 or duration > 600:
                print("Duration must be between 1 and 600 seconds. Using default 60.")
                duration = 60
        except ValueError:
            print("Invalid number. Using default 60.")
            duration = 60
    else:
        duration = 60

    commercial = input("Commercial use? (y/n): ").strip().lower() == 'y'

    user_input = {
        "prompt": prompt,
        "genre": genre,
        "mood": mood,
        "duration": duration,
        "commercial_use": commercial,
    }

    print("\n" + "=" * 60)
    print("Starting music generation workflow...")
    print("=" * 60 + "\n")

    try:
        supervisor = SupervisorAgent()
        result = await supervisor.run(user_input)

        if isinstance(result, dict):
            final_audio = result.get("final_audio")
            total_cost = result.get("total_cost", 0.0)
            fallback_used = result.get("fallback_used", False)
            creative_brief = result.get("creative_brief")
        else:
            final_audio = result.final_audio
            total_cost = result.total_cost
            fallback_used = result.fallback_used
            creative_brief = result.creative_brief

        # 最终兜底：如果没有生成任何音频，创建静音文件
        if final_audio is None or not Path(final_audio).exists():
            logger.warning("No valid audio produced. Creating silent fallback.")
            silent_path = Path(config.paths.output_dir) / f"silent_{int(asyncio.get_event_loop().time())}.wav"
            try:
                from pydub import AudioSegment
                silent = AudioSegment.silent(duration=duration * 1000)
                silent.export(str(silent_path), format="wav")
                final_audio = silent_path
                logger.info(f"Silent audio created at {silent_path}")
            except ImportError:
                logger.error("pydub not installed, cannot create silent audio.")
                final_audio = None

        print("\n" + "=" * 60)
        print("Generation Complete!")
        print("=" * 60)
        print(f"\nFinal audio saved to: {final_audio}")
        print(f"Total cost: ${total_cost:.4f}")

        if fallback_used:
            print("\n[Note] Fallback mode was used. Some services may have been unavailable.")

        if creative_brief:
            print("\nCreative Brief:")
            print(f"  - Genre: {creative_brief.get('genre')}")
            print(f"  - Mood: {creative_brief.get('mood')}")
            print(f"  - Duration: {creative_brief.get('duration')}s")
            print(f"  - Has Vocals: {creative_brief.get('has_vocals')}")

    except Exception as e:
        logger.exception(f"Workflow failed: {e}")
        print(f"\nError: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())