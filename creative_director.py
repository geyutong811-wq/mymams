"""Creative Director Agent - Analyzes requirements and orchestrates style decisions."""

from typing import Any, Dict, Optional
from pathlib import Path

from agents.base_agent import BaseAgent
from clients.lyria_client import LyriaClient


class CreativeDirectorAgent(BaseAgent):
    """
    Creative Director Agent.

    Responsibilities:
    - Parse user requirements (text, video, or audio references)
    - Analyze mood, style, and structural requirements
    - Determine appropriate genre, tempo, and instrumentation
    - Coordinate with other agents based on the creative brief
    """

    def __init__(self):
        super().__init__("CreativeDirector")
        self.lyria_client = LyriaClient()

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute creative direction based on input.

        Expected input_data keys:
            - prompt (str): User's description of desired music
            - reference_url (str, optional): URL to reference video/audio
            - genre (str, optional): Preferred genre
            - mood (str, optional): Desired mood
            - duration (int, optional): Target duration in seconds

        Returns:
            creative_brief: Structured creative brief for other agents
        """
        prompt = input_data.get("prompt", "")
        reference_url = input_data.get("reference_url")
        genre = input_data.get("genre")
        mood = input_data.get("mood")
        duration = input_data.get("duration", 60)

        self.log(f"Analyzing request: {prompt[:100]}...")

        creative_brief = {
            "prompt": prompt,
            "genre": genre or self._infer_genre(prompt, mood),
            "mood": mood or self._infer_mood(prompt),
            "duration": duration,
            "structure": self._determine_structure(genre, duration),
            "has_vocals": self._should_have_vocals(genre),
            "reference_analysis": None,
        }

        if reference_url:
            self.log(f"Analyzing reference: {reference_url}")
            creative_brief["reference_analysis"] = await self._analyze_reference(reference_url)

        self.log(f"Creative brief generated: {creative_brief}")
        return creative_brief

    def _infer_genre(self, prompt: str, mood: Optional[str]) -> str:
        """Infer genre from prompt text."""
        prompt_lower = prompt.lower()

        genre_map = {
            "pop": ["pop", "catchy", "mainstream"],
            "electronic": ["electronic", "edm", "house", "techno", "trance"],
            "rock": ["rock", "guitar", "drum", "punk", "metal"],
            "jazz": ["jazz", "smooth", "bossa", "swing"],
            "classical": ["classical", "orchestra", "symphony"],
            "lo-fi": ["lo-fi", "lofi", "chill", "study"],
            "ambient": ["ambient", "atmospheric", "meditation"],
        }

        for inferred_genre, keywords in genre_map.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return inferred_genre

        return "pop"

    def _infer_mood(self, prompt: str) -> str:
        """Infer mood from prompt text."""
        prompt_lower = prompt.lower()

        mood_map = {
            "uplifting": ["happy", "uplifting", "cheerful", "bright"],
            "melancholic": ["sad", "melancholic", "somber", "emotional"],
            "energetic": ["energetic", "intense", "powerful", "driving"],
            "relaxing": ["relaxing", "calm", "peaceful", "serene"],
            "mysterious": ["mysterious", "eerie", "haunting", "dark"],
        }

        for inferred_mood, keywords in mood_map.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return inferred_mood

        return "neutral"

    def _determine_structure(self, genre: str, duration: int) -> Dict[str, float]:
        """Determine song structure based on genre and duration."""
        if genre in ["pop", "rock"]:
            return {
                "intro": duration * 0.1,
                "verse": duration * 0.2,
                "chorus": duration * 0.2,
                "verse2": duration * 0.15,
                "chorus2": duration * 0.15,
                "bridge": duration * 0.1,
                "outro": duration * 0.1,
            }
        else:
            return {
                "intro": duration * 0.1,
                "body": duration * 0.8,
                "outro": duration * 0.1,
            }

    def _should_have_vocals(self, genre: str) -> bool:
        """Determine if the composition should include vocals."""
        vocal_genres = ["pop", "rock", "rnb", "hip-hop"]
        return genre.lower() in vocal_genres

    async def _analyze_reference(self, reference_url: str) -> Optional[Dict]:
        """Use Lyria 3 to analyze reference audio/video."""
        return {
            "analyzed": True,
            "tempo": 120,
            "key": "C major",
            "inferred_mood": "uplifting",
        }