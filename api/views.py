import json
import traceback
from pathlib import Path
from typing import Optional

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods


def _path_to_media_url(local_path: Optional[str]) -> Optional[str]:
    if not local_path:
        return None
    p = Path(local_path).resolve()
    media_root = Path(settings.MEDIA_ROOT).resolve()
    try:
        rel = p.relative_to(media_root)
        base = settings.MEDIA_URL.rstrip("/")
        return f"{base}/{rel.as_posix()}"
    except ValueError:
        return None


def _enrich_with_urls(payload: dict) -> dict:
    out = dict(payload)
    out["final_audio_url"] = _path_to_media_url(out.get("final_audio_path"))
    stems = out.get("stems")
    if stems:
        out["stems_urls"] = {k: _path_to_media_url(v) for k, v in stems.items()}
    return out


def _as_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("0", "false", "no", "off"):
        return False
    return True


def _parse_request_body(data: dict) -> dict:
    """将前端字段映射为 Supervisor.run 的 user_input（与 agents、cost_monitor、分轨逻辑对齐）。"""
    prompt = (data.get("prompt") or "").strip()

    style_id = (data.get("style_id") or "cyberpunk").strip().lower()
    style_to_genre = {
        "cyberpunk": "electronic",
        "epic": "rock",
        "lofi": "jazz",
        "commercial": "ambient",
    }
    genre = style_to_genre.get(style_id, "pop")
    commercial_use = style_id == "commercial" or _as_bool(
        data.get("commercial_use"), default=False
    )

    raw_dur = data.get("duration_seconds", data.get("duration", 180))
    try:
        duration = int(raw_dur)
    except (TypeError, ValueError):
        duration = 180
    duration = max(10, min(600, duration))
    duration = (duration // 10) * 10
    if duration < 10:
        duration = 10

    enable_vocal = _as_bool(data.get("enable_vocal_synthesis"), default=True)
    vocal_type = (data.get("vocal_voice_type") or "female").strip().lower()
    if vocal_type not in ("male", "female", "child"):
        vocal_type = "female"

    return {
        "prompt": prompt,
        "genre": genre,
        "mood": None,
        "duration": duration,
        "commercial_use": commercial_use,
        "enable_stem_separation": _as_bool(
            data.get("enable_stem_separation"), default=True
        ),
        "enable_vocal_synthesis": enable_vocal,
        "enable_cost_tracking": _as_bool(
            data.get("enable_cost_tracking"), default=True
        ),
        "vocal_voice_type": vocal_type if enable_vocal else None,
    }


@ensure_csrf_cookie
def index(request):
    return render(request, "index.html")


@require_http_methods(["POST"])
def generate(request):
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)

    user_input = _parse_request_body(body)
    if not user_input["prompt"]:
        return JsonResponse({"ok": False, "error": "Prompt 不能为空"}, status=400)

    try:
        from main import run_generation_sync

        result = run_generation_sync(user_input)
        result = _enrich_with_urls(result)
        return JsonResponse({"ok": True, "result": result})
    except Exception as e:
        traceback.print_exc()
        return JsonResponse(
            {"ok": False, "error": str(e), "detail": traceback.format_exc()},
            status=500,
        )
