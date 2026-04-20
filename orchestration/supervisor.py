"""Supervisor Agent - Orchestrates multi-agent workflows using LangGraph."""

import asyncio
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from agents.creative_director import CreativeDirectorAgent
from agents.lyrics_agent import LyricsAgent
from agents.melody_agent import MelodyAgent
from agents.accompaniment_agent import AccompanimentAgent
from agents.atmosphere import AtmosphereAgent
from agents.commercial_bgm import CommercialBGMAgent
from agents.stem_separation import StemSeparationAgent
from agents.research import ResearchAgent
from audio.audio_mixer import AudioMixer
from config import config
from cost_monitor import CostMonitor
from utils.logger import setup_logger

logger = setup_logger("Supervisor", log_file=Path(config.paths.output_dir) / "supervisor.log")


class WorkflowState(BaseModel):
    """State model for the LangGraph workflow."""
    user_prompt: str
    user_reference_url: Optional[str] = None
    user_genre: Optional[str] = None
    user_mood: Optional[str] = None
    user_duration: int = 60
    commercial_use: bool = False

    enable_stem_separation: bool = True
    enable_vocal_synthesis: bool = True
    vocal_voice_type: Optional[str] = None

    creative_brief: Optional[Dict[str, Any]] = None
    lyrics: Optional[str] = None

    main_audio: Optional[Path] = None
    atmospheric_audio: Optional[Path] = None
    bgm_audio: Optional[Path] = None
    stems: Optional[Dict[str, Path]] = None

    final_audio: Optional[Path] = None

    total_cost: float = 0.0

    current_step: str = "init"
    error: Optional[str] = None
    fallback_used: bool = False


class SupervisorAgent:
    _semaphore = asyncio.Semaphore(5)

    def __init__(self):
        self.creative_director = CreativeDirectorAgent()
        self.lyrics_agent = LyricsAgent()
        self.melody_agent = MelodyAgent()
        self.accompaniment_agent = AccompanimentAgent()
        self.atmosphere = AtmosphereAgent()
        self.commercial_bgm = CommercialBGMAgent()
        self.stem_separation = StemSeparationAgent()
        self.research = ResearchAgent()
        self.audio_mixer = AudioMixer()
        self.cost_monitor = CostMonitor()

        self.workflow = self._build_workflow()
        logger.info("SupervisorAgent initialized")

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(WorkflowState)

        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("generate_lyrics", self._generate_lyrics_node)
        workflow.add_node("generate_melody", self._generate_melody_node)
        workflow.add_node("generate_accompaniment", self._generate_accompaniment_node)
        workflow.add_node("generate_atmosphere", self._generate_atmosphere_node)
        workflow.add_node("generate_bgm", self._generate_bgm_node)
        workflow.add_node("separate_stems", self._separate_stems_node)
        workflow.add_node("mix_final", self._mix_final_node)
        workflow.add_node("handle_error", self._error_node)

        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "generate_lyrics")
        workflow.add_edge("generate_lyrics", "generate_melody")
        workflow.add_edge("generate_melody", "generate_accompaniment")

        # 可选增强节点（氛围、BGM、分轨）放在伴奏之后、混音之前
        workflow.add_conditional_edges(
            "generate_accompaniment",
            self._route_after_accompaniment,
            {
                "atmosphere": "generate_atmosphere",
                "bgm": "generate_bgm",
                "stems": "separate_stems",
                "mix": "mix_final",
                "error": "handle_error",
            }
        )

        workflow.add_conditional_edges(
            "generate_atmosphere",
            self._route_after_atmosphere,
            {
                "bgm": "generate_bgm",
                "stems": "separate_stems",
                "mix": "mix_final",
                "error": "handle_error",
            }
        )
        workflow.add_conditional_edges(
            "generate_bgm",
            self._route_after_bgm,
            {
                "stems": "separate_stems",
                "mix": "mix_final",
                "error": "handle_error",
            }
        )
        workflow.add_conditional_edges(
            "separate_stems",
            self._route_after_stems,
            {
                "mix": "mix_final",
                "error": "handle_error",
            }
        )

        workflow.add_edge("mix_final", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile(checkpointer=MemorySaver())

    def _route_after_accompaniment(self, state: WorkflowState) -> Literal["atmosphere", "bgm", "stems", "mix", "error"]:
        if state.error:
            return "error"
        brief = state.creative_brief or {}
        genre = brief.get("genre", "")
        has_vocals = brief.get("has_vocals", False)
        if has_vocals:
            return "stems" if state.enable_stem_separation else "mix"
        elif state.commercial_use:
            return "bgm"
        elif genre not in ["ambient", "lo-fi", "classical"]:
            return "atmosphere"
        else:
            return "mix"

    def _route_after_atmosphere(self, state: WorkflowState) -> Literal["bgm", "stems", "mix", "error"]:
        if state.error:
            return "error"
        brief = state.creative_brief or {}
        has_vocals = brief.get("has_vocals", False)
        if has_vocals:
            return "stems" if state.enable_stem_separation else "mix"
        elif state.commercial_use:
            return "bgm"
        else:
            return "mix"

    def _route_after_bgm(self, state: WorkflowState) -> Literal["stems", "mix", "error"]:
        if state.error:
            return "error"
        brief = state.creative_brief or {}
        if brief.get("has_vocals", False):
            return "stems" if state.enable_stem_separation else "mix"
        else:
            return "mix"

    def _route_after_stems(self, state: WorkflowState) -> Literal["mix", "error"]:
        if state.error:
            return "error"
        return "mix"

    async def _analyze_node(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 1: Analyzing user request...")
        try:
            creative_brief = await asyncio.wait_for(
                self.creative_director.execute({
                    "prompt": state.user_prompt,
                    "reference_url": state.user_reference_url,
                    "genre": state.user_genre,
                    "mood": state.user_mood,
                    "duration": state.user_duration,
                }),
                timeout=30.0
            )
            state.creative_brief = creative_brief
            if not state.enable_vocal_synthesis:
                creative_brief["has_vocals"] = False
            if state.vocal_voice_type and state.enable_vocal_synthesis:
                creative_brief["vocal_voice_type"] = state.vocal_voice_type
            state.current_step = "analyzed"
        except asyncio.TimeoutError:
            state.error = "Analysis timeout"
            logger.error(state.error)
        except Exception as e:
            state.error = f"Analysis failed: {e}"
            logger.error(state.error)
        return state

    async def _generate_lyrics_node(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 2: Generating lyrics...")
        if state.error:
            return state
        brief = state.creative_brief or {}
        try:
            result = await asyncio.wait_for(
                self.lyrics_agent.execute({
                    "prompt": brief.get("prompt"),
                    "genre": brief.get("genre"),
                    "mood": brief.get("mood"),
                    "duration": brief.get("duration"),
                }),
                timeout=60.0
            )
            state.lyrics = result.get("lyrics")
            self.cost_monitor.add_cost("lyrics_generation", 0.001, success=True)
            state.current_step = "lyrics_generated"
        except Exception as e:
            state.error = f"Lyrics generation failed: {e}"
            logger.error(state.error)
        return state

    async def _generate_melody_node(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 3: Generating melody (vocals)...")
        if state.error:
            return state
        brief = state.creative_brief or {}
        try:
            result = await asyncio.wait_for(
                self.melody_agent.execute({
                    "lyrics": state.lyrics,
                    "genre": brief.get("genre"),
                    "mood": brief.get("mood"),
                    "duration": brief.get("duration"),
                    "prompt": brief.get("prompt"),
                }),
                timeout=180.0
            )
            state.main_audio = result.get("main_audio")
            success = state.main_audio is not None
            self.cost_monitor.add_cost("melody_generation", 0.02, success=success)
            state.current_step = "melody_generated"
        except Exception as e:
            state.error = f"Melody generation failed: {e}"
            logger.error(state.error)
        return state

    async def _generate_accompaniment_node(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 4: Generating accompaniment...")
        if state.error:
            return state
        brief = state.creative_brief or {}
        try:
            result = await asyncio.wait_for(
                self.accompaniment_agent.execute({
                    "genre": brief.get("genre"),
                    "mood": brief.get("mood"),
                    "duration": brief.get("duration"),
                    "lyrics": state.lyrics,
                }),
                timeout=120.0
            )
            state.bgm_audio = result.get("accompaniment_audio")
            success = state.bgm_audio is not None
            self.cost_monitor.add_cost("accompaniment_generation", 0.01, success=success)
            state.current_step = "accompaniment_generated"
        except Exception as e:
            state.error = f"Accompaniment generation failed: {e}"
            logger.error(state.error)
        return state

    async def _generate_atmosphere_node(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 5 (optional): Generating atmosphere...")
        if state.error:
            return state
        brief = state.creative_brief or {}
        try:
            result = await asyncio.wait_for(
                self.atmosphere.execute({
                    "prompt": f"Ambient background for {brief.get('genre')} music, {brief.get('mood')} mood",
                    "duration": brief.get("duration", 60),
                }),
                timeout=60.0
            )
            state.atmospheric_audio = result.get("audio_path")
            success = state.atmospheric_audio is not None
            self.cost_monitor.add_cost("atmosphere", 0.02, success=success)
            state.current_step = "atmosphere_generated"
        except Exception as e:
            logger.error(f"Atmosphere generation failed (non-critical): {e}")
        return state

    async def _generate_bgm_node(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 5 (optional): Generating BGM...")
        if state.error:
            return state
        brief = state.creative_brief or {}
        try:
            result = await asyncio.wait_for(
                self.commercial_bgm.execute({
                    "genre": brief.get("genre", "electronic"),
                    "mood": brief.get("mood", "uplifting"),
                    "duration": brief.get("duration", 60),
                    "use_case": "video_background" if state.commercial_use else "general",
                }),
                timeout=60.0
            )
            state.bgm_audio = result.get("audio_path")
            success = state.bgm_audio is not None
            self.cost_monitor.add_cost("commercial_bgm", 0.01, success=success)
            state.current_step = "bgm_generated"
        except Exception as e:
            logger.error(f"BGM generation failed (non-critical): {e}")
        return state

    async def _separate_stems_node(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 5 (optional): Separating stems...")
        if not state.enable_stem_separation:
            state.current_step = "stems_skipped"
            return state
        if state.error or not state.main_audio:
            return state
        try:
            result = await asyncio.wait_for(
                self.stem_separation.execute({
                    "audio_path": state.main_audio,
                    "stems_config": "4stems",
                }),
                timeout=180.0
            )
            state.stems = result.get("stem_paths", {})
            self.cost_monitor.add_cost("stem_separation", 0.0, success=True)
            state.current_step = "stems_separated"
        except Exception as e:
            logger.error(f"Stem separation failed (non-critical): {e}")
        return state

    async def _mix_final_node(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 6: Mixing final audio...")
        if state.error:
            return await self._error_node(state)

        final_audio = state.main_audio
        if not final_audio:
            state.error = "No main audio to mix"
            return await self._error_node(state)

        # 混音：优先使用 stems 重建（如果有）
        if state.stems and len(state.stems) > 0:
            vocal_gain = getattr(config.mixing, 'vocal_gain', 3)
            drum_gain = getattr(config.mixing, 'drum_gain', -3)
            bass_gain = getattr(config.mixing, 'bass_gain', -2)
            other_gain = getattr(config.mixing, 'other_gain', -6)
            stem_volumes = []
            for stem_name, stem_path in state.stems.items():
                # ========================================================
                # 【修改点：去除鼓点 (Drumless) 逻辑】
                # 如果当前音轨是鼓点 (drums)，则直接跳过，不加入最终的混音列表
                if stem_name == "drums":
                    logger.info("Drumless mode: Skipping drums stem in final mix.")
                    continue
                # ========================================================
                if stem_name == "vocals":
                    vol = vocal_gain
                elif stem_name == "drums":
                    vol = drum_gain
                elif stem_name == "bass":
                    vol = bass_gain
                else:
                    vol = other_gain
                stem_volumes.append((stem_path, vol))
            try:
                stem_mix = await self.audio_mixer.mix_stems(stem_volumes)
                final_audio = stem_mix
            except Exception as e:
                logger.error(f"Stem mixing failed: {e}")

        # 叠加氛围音
        if state.atmospheric_audio:
            try:
                final_audio = await self.audio_mixer.overlay_audio(
                    final_audio, state.atmospheric_audio,
                    volume_adjustment_db=-12
                )
            except Exception as e:
                logger.error(f"Atmosphere overlay failed: {e}")

        # 叠加 BGM（如果有伴奏且不是 stems 重建时）
        if state.bgm_audio and not state.stems:
            try:
                final_audio = await self.audio_mixer.overlay_audio(
                    final_audio, state.bgm_audio,
                    volume_adjustment_db=-8
                )
            except Exception as e:
                logger.error(f"BGM overlay failed: {e}")

        # 后处理
        if final_audio:
            try:
                final_audio = await self.audio_mixer.apply_fade(
                    final_audio, fade_in_ms=1000, fade_out_ms=2000
                )
                final_audio = await self.audio_mixer.normalize_loudness(final_audio)
            except Exception as e:
                logger.error(f"Post-processing failed: {e}")

        # 统一输出格式，并保存到 output_dir
        if final_audio:
            try:
                # 使用 save_as_format 并指定输出目录为 output_dir
                output_filename = f"final_{int(asyncio.get_event_loop().time())}.{config.output_format}"
                output_path = Path(config.paths.output_dir) / output_filename
                final_audio = await self.audio_mixer.save_as_format(final_audio, output_path=output_path)
                logger.info(f"Final audio saved to {final_audio}")
            except Exception as e:
                logger.error(f"Format conversion failed: {e}")

        state.final_audio = final_audio
        state.total_cost = self.cost_monitor.get_total_cost()
        state.current_step = "finalized"
        logger.info(f"Workflow complete. Final audio: {state.final_audio}")
        logger.info(f"Total cost: ${state.total_cost:.4f}")
        return state

    async def _error_node(self, state: WorkflowState) -> WorkflowState:
        logger.error(f"Handling error: {state.error}")
        if config.enable_fallback and config.fallback_to_pure_instrumental:
            logger.info("Fallback: generating pure instrumental...")
            try:
                result = await asyncio.wait_for(
                    self.commercial_bgm.execute({
                        "genre": "ambient",
                        "mood": "neutral",
                        "duration": state.user_duration,
                    }),
                    timeout=60.0
                )
                state.final_audio = result.get("audio_path")
                state.fallback_used = True
            except Exception as e:
                logger.error(f"Fallback also failed: {e}")
                state.final_audio = None
        else:
            state.final_audio = None

        # 最终兜底：生成静音文件并保存到 output_dir
        if state.final_audio is None:
            logger.warning("All generation attempts failed. Creating silent audio as final fallback.")
            silent_path = Path(config.paths.output_dir) / f"silent_{int(asyncio.get_event_loop().time())}.wav"
            silent_path.parent.mkdir(parents=True, exist_ok=True)
            from pydub import AudioSegment
            silent = AudioSegment.silent(duration=state.user_duration * 1000)
            silent.export(str(silent_path), format="wav")
            state.final_audio = silent_path
            state.fallback_used = True

        state.current_step = "error_handled"
        return state

    async def run(self, user_input: Dict[str, Any]) -> WorkflowState:
        async with self._semaphore:
            prompt = user_input.get("prompt", "").strip()
            if not prompt:
                raise ValueError("Prompt cannot be empty")
            duration = int(user_input.get("duration", 60))
            if duration <= 0 or duration > 600:
                raise ValueError("Duration must be between 1 and 600 seconds")

            self.cost_monitor.reset()
            self.cost_monitor.set_enabled(user_input.get("enable_cost_tracking", True))

            initial_state = WorkflowState(
                user_prompt=prompt,
                user_reference_url=user_input.get("reference_url"),
                user_genre=user_input.get("genre"),
                user_mood=user_input.get("mood"),
                user_duration=duration,
                commercial_use=user_input.get("commercial_use", False),
                enable_stem_separation=bool(user_input.get("enable_stem_separation", True)),
                enable_vocal_synthesis=bool(user_input.get("enable_vocal_synthesis", True)),
                vocal_voice_type=user_input.get("vocal_voice_type"),
            )

            configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
            final_state = await self.workflow.ainvoke(initial_state, config=configurable)

            if not getattr(config, 'keep_temp_files', False):
                await self._cleanup_temp_files_async(final_state)
            return final_state

    async def _cleanup_temp_files_async(self, state: WorkflowState):
        """异步删除临时目录，避免阻塞事件循环。"""
        temp_dir = Path(config.paths.temp_dir) if config.paths.temp_dir else None
        stems_dir = Path(config.paths.stems_dir) if config.paths.stems_dir else None
        temp_dirs = [d for d in [temp_dir, stems_dir] if d is not None]
        for d in temp_dirs:
            if d.exists():
                try:
                    await asyncio.to_thread(shutil.rmtree, str(d))
                    await asyncio.to_thread(d.mkdir, parents=True, exist_ok=True)
                    logger.info(f"Cleaned up temporary directory: {d}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {d}: {e}")