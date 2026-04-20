"""Stem Separation Agent - Splits audio into constituent parts."""
from typing import Any, Dict, Optional
from pathlib import Path

from agents.base_agent import BaseAgent
from audio.stem_separator import StemSeparator
from audio.audio_mixer import AudioMixer


class StemSeparationAgent(BaseAgent):
    """
    Stem Separation Agent.
    
    Responsibilities:
    - Split audio into vocals, accompaniment, drums, bass
    - Enable remixing and post-processing
    - Extract specific stems for further processing
    
    Tools:
    - Spleeter: Open-source audio separation library
    - AudioMixer: For reassembling stems
    """
    
    def __init__(self):
        super().__init__("StemSeparation")
        self.separator = StemSeparator(stems="4stems")
        self.mixer = AudioMixer()
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Separate audio into stems.
        
        Expected input_data keys:
            - audio_path (Path): Path to input audio
            - stems_config (str): "2stems", "4stems", or "5stems"
            - extract_vocals_only (bool): Only extract vocals
            
        Returns:
            stem_paths: Dictionary mapping stem names to file paths
            original_path: Original audio path
        """
        audio_path = input_data.get("audio_path")
        stems_config = input_data.get("stems_config", "4stems")
        extract_vocals_only = input_data.get("extract_vocals_only", False)
        
        if not audio_path:
            self.log("No audio_path provided")
            return {"error": "audio_path required"}
        
        # Reconfigure separator if needed
        if stems_config != self.separator.stems:
            self.separator = StemSeparator(stems=stems_config)
        
        self.log(f"Separating {audio_path} into {stems_config}")
        
        if extract_vocals_only:
            vocal_path = await self.separator.get_vocals(audio_path)
            result = {
                "stem_paths": {"vocals": vocal_path},
                "original_path": audio_path,
            }
        else:
            stem_paths = await self.separator.separate(audio_path)
            result = {
                "stem_paths": stem_paths,
                "original_path": audio_path,
            }
        
        self.log(f"Separation complete. Stems: {list(result['stem_paths'].keys())}")
        return result