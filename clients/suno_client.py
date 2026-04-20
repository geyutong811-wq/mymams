"""Lyria 3 Client - Vertex AI using PredictionServiceClient (corrected)."""

import asyncio
import time
import base64
from pathlib import Path
from typing import Optional
from google.cloud import aiplatform_v1beta1 as aiplatform
from google.oauth2 import service_account

from config import config


class LyriaClient:
    """Corrected Lyria client using Vertex AI PredictionServiceClient."""

    def __init__(self):
        self.project_id = config.api.vertex_project_id
        self.location = config.api.vertex_location
        self.credentials_path = config.api.google_credentials_path
        if not self.project_id:
            raise ValueError("VERTEX_AI_PROJECT_ID missing")
        if self.credentials_path:
            import os
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            client_options = {"api_endpoint": f"{self.location}-aiplatform.googleapis.com"}
            self.client = aiplatform.PredictionServiceClient(client_options=client_options)
        except Exception as e:
            # 修复：抛出异常，避免静默失败
            raise RuntimeError(f"Failed to initialize Lyria client: {e}")

    async def generate(
        self,
        prompt: str,
        duration: int = 30,
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """Generate music using Lyria 3."""
        if not self.client:
            raise RuntimeError("Lyria client not initialized")
        endpoint = f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/lyria-002"
        instance = {"prompt": prompt, "duration": duration}

        def _sync_predict():
            return self.client.predict(endpoint=endpoint, instances=[instance])

        try:
            response = await asyncio.to_thread(_sync_predict)
            audio_b64 = response.predictions[0].get("audio")
            if not audio_b64:
                return None
            audio_bytes = base64.b64decode(audio_b64)
            if output_path is None:
                output_path = Path(config.paths.output_dir) / f"lyria_{int(time.time())}.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            return output_path
        except Exception as e:
            print(f"[Lyria] Generation failed: {e}")
            return None