"""Browser-accessible TalkTracker server mode.

The server deliberately keeps a single analysis worker.  A workstation that
has the local Whisper and Nemotron models can therefore serve recordings from
other PCs without sharing its model files or GPU with concurrent jobs.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import shutil
import sys
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from .core.jobs import JobManager, JobNotFoundError, JobStateError
from .core.llm_client import LocalLLMError
from .core.workflow import LocalLLMConnectionManager
from .core.models import (
    WHISPER_PROFILES,
    WhisperProfile,
    get_cuda_runtime_status,
    get_whisper_runtime,
    missing_whisper_runtime_files,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_MAX_UPLOAD_MB = 2048
DEFAULT_RETENTION_DAYS = 7
UPLOAD_CHUNK_BYTES = 1024 * 1024


async def _read_local_llm_request(request: Request, *, require_model: bool) -> tuple[str, str, str]:
    """Read transient connection input without reflecting API keys in errors."""

    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="로컬 AI 연결 입력 형식이 올바르지 않습니다.") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="로컬 AI 연결 입력 형식이 올바르지 않습니다.")

    base_url = payload.get("base_url", "")
    api_key = payload.get("api_key", "")
    model = payload.get("model", "")
    if not isinstance(base_url, str) or not isinstance(api_key, str) or not isinstance(model, str):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="로컬 AI 연결 입력 형식이 올바르지 않습니다.")
    if require_model and not model.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="연결을 확인할 모델 ID를 입력하거나 선택해 주세요.")
    return base_url, api_key, model


def _integer_setting(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} 값은 정수여야 합니다.") from error
    if value < minimum:
        raise ValueError(f"{name} 값은 {minimum} 이상이어야 합니다.")
    return value


def _is_loopback_host(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "::1", "localhost"}


def _native_nemotron_binary(project_root: Path) -> Path | None:
    """Find the native diarization runtime installed for macOS or Linux."""

    configured = os.environ.get("TALKTRACKER_NEMO_SPEECH")
    candidates = [
        Path(configured).expanduser() if configured else None,
        project_root / ".tools" / "nemo-speech" / "bin" / "nemo-speech",
    ]
    path_binary = shutil.which("nemo-speech")
    if path_binary:
        candidates.append(Path(path_binary))
    for candidate in candidates:
        if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _model_download_command(profile: WhisperProfile) -> str:
    """Return the current platform's command for preparing one profile.

    Windows downloads the upstream checkpoint first, so it must also request
    its local CTranslate2 conversion.  macOS and Linux download an already
    prepared model directly.
    """

    if sys.platform == "win32":
        return (
            "powershell -ExecutionPolicy Bypass -File "
            f".\\scripts\\windows\\Download-Models.ps1 -WhisperProfile {profile.key} "
            "-PrepareForLocalUse"
        )
    if sys.platform == "darwin":
        return f"bash scripts/macos/Download-Models.sh --whisper-profile {profile.key}"
    return f"bash scripts/linux/Download-Models.sh --whisper-profile {profile.key}"


def _profile_availability(project_root: Path, profile: WhisperProfile) -> tuple[bool, str | None]:
    """Return whether a profile can start without a model-related failure."""

    runtime = get_whisper_runtime(profile)
    if not runtime:
        return False, "호환 CUDA GPU가 없어 이 전사 모델을 실행할 수 없습니다."

    missing_files = missing_whisper_runtime_files(project_root / "models" / runtime.folder, profile)
    if missing_files:
        command = _model_download_command(profile)
        return False, f"Whisper 모델이 준비되지 않았습니다 ({', '.join(missing_files)} 없음). {command}을 실행해 주세요."

    native_diarization = sys.platform == "darwin" or sys.platform.startswith("linux")
    nemotron_weight = "Nemotron-3-Diarization.q8_0.gguf" if native_diarization else "model.safetensors"
    if not (project_root / "models" / "nemotron-3-diarization" / nemotron_weight).is_file():
        return False, "Nemotron 화자 분리 모델이 준비되지 않았습니다. 모델 다운로드 스크립트를 다시 실행해 주세요."
    if native_diarization and not _native_nemotron_binary(project_root):
        return False, "Nemotron 화자 분리 실행기가 준비되지 않았습니다. Install-Dependencies 스크립트를 다시 실행해 주세요."
    return True, None


def _profile_payload(project_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "key": profile.key,
            "label": profile.label,
            "description": profile.description,
            "requires_gpu": profile.requires_gpu,
            "available": (availability := _profile_availability(project_root, profile))[0],
            "unavailable_reason": availability[1],
        }
        for profile in WHISPER_PROFILES
    ]


def _server_unavailable_page() -> str:
    """Show a distinct status page when this PC cannot process any request."""

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TalkTracker 서버 · 준비 필요</title>
  <style>
    :root {{ color-scheme: light; --navy:#103640; --teal:#147d7e; --mint:#e5f4ef; --line:#d8e6e2; --ink:#193037; --muted:#617178; --danger:#a64242; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:#f4f7f6; color:var(--ink); font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ padding:34px 22px; color:#fff; background:var(--navy); }} header div, main {{ width:min(800px,calc(100% - 32px)); margin:auto; }}
    h1 {{ margin:0; font-size:29px; }} header p {{ margin:6px 0 0; color:#cde5df; }} main {{ padding:34px 0 52px; }}
    section {{ padding:28px; border:1px solid var(--line); border-radius:12px; background:#fff; box-shadow:0 4px 14px rgb(16 54 64 / 5%); }}
    .badge {{ display:inline-block; padding:3px 10px; border-radius:999px; color:#923b3b; background:#fbecec; font-size:13px; font-weight:750; }} h2 {{ margin:13px 0 8px; font-size:25px; }} h3 {{ margin:26px 0 7px; font-size:17px; }} p {{ margin:8px 0; }} .reason {{ margin-top:20px; padding:13px 15px; border-radius:8px; color:var(--danger); background:#fbecec; }} .reason p {{ margin:3px 0 0; }} .hint {{ color:var(--muted); }} code {{ overflow-wrap:anywhere; padding:2px 5px; border-radius:4px; background:#edf3f1; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.9em; }}
  </style>
</head>
<body>
<header><div><h1>TalkTracker 서버</h1><p>이 서버 PC에서 화자 분리와 전사를 처리합니다.</p></div></header>
<main><section>
  <span class="badge">분석 요청 수신 불가</span>
  <h2>이 PC는 현재 요청 처리 서버가 아닙니다.</h2>
  <p>이 PC에는 분석에 필요한 로컬 모델이 준비되지 않아, 이 서버는 분석 요청을 받을 수 없습니다.</p>
  <div class="reason"><strong>현재 상태</strong><p>실행 가능한 전사 프로필이 없습니다. 파일 업로드와 분석 요청은 받지 않습니다.</p></div>
  <h3>이 PC에서 분석 서버를 운영하려면</h3>
  <p>로컬 모델을 준비한 뒤 웹 서버를 다시 시작하세요.</p>
  <h3>다른 PC의 분석 서버를 사용하려면</h3>
  <p class="hint">이 PC에서는 <code>Run-Web</code>을 실행하지 말고, 모델과 처리 성능이 있는 서버 PC의 웹 주소로 브라우저에서 접속하세요.</p>
</section></main>
</body></html>"""


def _index_page(*, access_token_required: bool = False, local_llm_available: bool = True) -> str:
    """Keep the initial client self-contained; no frontend build is required."""

    page = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TalkTracker 서버</title>
  <style>
    :root { color-scheme: light; --navy:#103640; --teal:#147d7e; --mint:#e5f4ef; --line:#d8e6e2; --ink:#193037; --muted:#617178; --danger:#a64242; }
    * { box-sizing: border-box; } body { margin:0; background:#f4f7f6; color:var(--ink); font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    header { padding:34px 22px; color:#fff; background:var(--navy); } header div, main { width:min(920px,calc(100% - 32px)); margin:auto; }
    h1 { margin:0; font-size:29px; } header p { margin:6px 0 0; color:#cde5df; } main { padding:26px 0 48px; }
    section { margin-bottom:16px; padding:22px; border:1px solid var(--line); border-radius:12px; background:#fff; box-shadow:0 4px 14px rgb(16 54 64 / 5%); }
    h2 { margin:0 0 14px; font-size:19px; } label { display:block; margin:12px 0 5px; font-weight:650; }
    input, select, button { min-height:38px; border:1px solid #aac4bf; border-radius:7px; font:inherit; } input, select { width:100%; padding:7px 9px; background:#fff; }
    button { padding:7px 15px; cursor:pointer; color:#fff; border-color:var(--teal); background:var(--teal); font-weight:700; }
    button.secondary { color:var(--ink); background:#fff; border-color:#a9bbb7; } button.danger { color:#fff; background:var(--danger); border-color:var(--danger); }
    button:disabled { opacity:.55; cursor:wait; } .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; } .actions { display:flex; gap:9px; align-items:center; margin-top:17px; }
    .hint { color:var(--muted); font-size:13px; } .notice { padding:10px 12px; border-radius:8px; background:var(--mint); color:#1e6655; } .error { color:var(--danger); background:#fbecec; }
    .hidden { display:none; } .progress { height:10px; overflow:hidden; border-radius:999px; background:#e6efed; } .progress > div { height:100%; width:0; background:var(--teal); transition:width .2s; }
    dl { display:grid; grid-template-columns:126px 1fr; gap:6px 14px; margin:16px 0 0; } dt { color:var(--muted); } dd { margin:0; word-break:break-word; }
    .timeline { margin-top:18px; padding:14px; border:1px solid var(--line); border-radius:9px; background:#f9fbfa; overflow-x:auto; } .timeline-title { margin:0 0 9px; font-size:13px; font-weight:750; } .timeline-scale { display:flex; justify-content:space-between; margin:0 3px 5px 88px; color:var(--muted); font-size:11px; } .speaker-lane { display:grid; grid-template-columns:76px minmax(420px, 1fr); gap:10px; align-items:center; min-width:560px; margin:5px 0; } .speaker-label { overflow:hidden; color:var(--muted); font-size:12px; font-weight:700; text-overflow:ellipsis; white-space:nowrap; } .speaker-track { position:relative; height:14px; border-radius:4px; background:repeating-linear-gradient(90deg,#e8efed 0,#e8efed calc(10% - 1px),#dfe9e6 calc(10% - 1px),#dfe9e6 10%); } .speaker-turn { position:absolute; top:2px; height:10px; min-width:2px; border-radius:3px; box-shadow:0 1px 1px rgb(16 54 64 / 18%); } .timeline-empty { color:var(--muted); font-size:13px; }
    a { color:#096a71; font-weight:700; } .job { padding:10px 0; border-top:1px solid var(--line); } .job:first-child { border-top:0; }
    @media (max-width:620px) { .grid { grid-template-columns:1fr; } dl { grid-template-columns:1fr; gap:1px; } dt { margin-top:8px; } .timeline { padding:11px; } }
  </style>
</head>
<body>
<header><div><h1>TalkTracker 서버</h1><p>이 서버 PC에서 화자 분리와 전사를 처리합니다.</p></div></header>
<main>
  __LOCAL_LLM_SECTION__
  <section>
    <h2>새 분석 요청</h2>
    <p class="hint">지원 형식: M4A, MP3, WAV, FLAC, AAC, OGG · 서버는 한 번에 하나의 작업만 처리합니다.</p>
    <form id="upload-form">
      <div class="grid"><div><label for="audio">음성 파일</label><input id="audio" name="file" type="file" accept=".m4a,.mp3,.wav,.flac,.aac,.ogg" required></div>
      <div><label for="profile">전사 품질</label><select id="profile" name="profile" disabled></select><p id="profile-status" class="hint">서버의 준비된 모델을 확인하고 있습니다…</p></div></div>
      __ACCESS_TOKEN_SECTION__
      <div class="actions"><button id="submit" type="submit" disabled>업로드하고 분석 시작</button><span id="upload-status" class="hint"></span></div>
    </form>
    <div id="server-setup" class="notice error hidden" role="status"></div>
  </section>
  <section id="current" class="hidden">
    <h2>현재 작업</h2><div id="current-notice" class="notice"></div><div class="progress" aria-label="분석 진행률"><div id="progress-bar"></div></div>
    <dl id="details"></dl><div id="timeline" class="timeline"></div><div class="actions"><button id="cancel" class="danger" type="button">중단</button><a id="result" class="hidden" download>Markdown 결과 다운로드</a></div>
  </section>
  <section><h2>이 서버의 최근 작업</h2><div id="jobs" class="hint">불러오는 중…</div></section>
</main>
<script>
  let activeId = null, events = null, polling = null, profiles = [];
  const $ = (id) => document.getElementById(id);
  const escapeHtml = (text) => String(text ?? "").replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
  const token = () => $("token")?.value.trim() || "";
  const authHeaders = () => token() ? {"X-TalkTracker-Token": token()} : {};
  const apiUrl = (path, includeToken = false) => includeToken && token() ? `${path}${path.includes("?") ? "&" : "?"}access_token=${encodeURIComponent(token())}` : path;
  async function request(path, options = {}) {
    const headers = new Headers(options.headers || {}); Object.entries(authHeaders()).forEach(([k,v]) => headers.set(k,v));
    const response = await fetch(apiUrl(path), {...options, headers});
    if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `요청에 실패했습니다 (${response.status}).`); }
    return response.status === 204 ? null : response.json();
  }
  function duration(value) { if (value == null) return "-"; const total = Math.max(0, Math.round(value)); return `${String(Math.floor(total / 60)).padStart(2,"0")}:${String(total % 60).padStart(2,"0")}`; }
  function render(job) {
    activeId = job.id; $("current").classList.remove("hidden"); $("current-notice").textContent = job.error || job.message; $("current-notice").classList.toggle("error", job.state === "failed");
    $("progress-bar").style.width = `${job.progress}%`;
    $("details").innerHTML = `<dt>파일</dt><dd>${escapeHtml(job.source_name)}</dd><dt>상태</dt><dd>${escapeHtml(job.state)}${job.queue_position ? ` · 대기 ${job.queue_position}번` : ""}</dd><dt>전사 품질</dt><dd>${escapeHtml(job.whisper_profile)}</dd><dt>진행률</dt><dd>${job.progress}%</dd><dt>녹음 길이</dt><dd>${duration(job.duration_seconds)}</dd>`;
    renderTimeline(job.speaker_segments || [], job.duration_seconds);
    const terminal = ["completed","cancelled","failed"].includes(job.state); $("cancel").disabled = terminal || job.cancel_requested;
    const result = $("result"); result.classList.toggle("hidden", !job.result_available); result.href = apiUrl(`/api/jobs/${job.id}/result.md`, true);
    if (terminal) stopWatching(); loadJobs();
  }
  const SPEAKER_COLORS = ["#58aa9b", "#e8a05a", "#877bd7", "#62a9c4", "#d37f98", "#91ad58", "#aa87c9", "#be9364"];
  function renderTimeline(segments, durationSeconds) {
    const timeline = $("timeline");
    const valid = segments.filter((turn) => Number.isFinite(Number(turn.start)) && Number.isFinite(Number(turn.end)) && Number(turn.end) > Number(turn.start));
    if (!valid.length) { timeline.innerHTML = '<p class="timeline-empty">화자 분리가 완료되면 발화 구간을 시간축으로 표시합니다.</p>'; return; }
    const durationValue = Math.max(Number(durationSeconds) || 0, ...valid.map((turn) => Number(turn.end)), 0.1);
    const speakers = [...new Set(valid.map((turn) => String(turn.speaker)))];
    const rows = speakers.map((speaker, index) => {
      const turns = valid.filter((turn) => String(turn.speaker) === speaker).map((turn) => {
        const start = Math.max(0, Math.min(durationValue, Number(turn.start)));
        const end = Math.max(start, Math.min(durationValue, Number(turn.end)));
        const left = (start / durationValue) * 100;
        const width = Math.max(0.3, ((end - start) / durationValue) * 100);
        const description = `${speaker} · ${duration(start)}–${duration(end)}`;
        return `<span class="speaker-turn" title="${escapeHtml(description)}" aria-label="${escapeHtml(description)}" style="left:${left.toFixed(3)}%;width:${width.toFixed(3)}%;background:${SPEAKER_COLORS[index % SPEAKER_COLORS.length]}"></span>`;
      }).join("");
      return `<div class="speaker-lane"><div class="speaker-label">${escapeHtml(speaker)}</div><div class="speaker-track">${turns}</div></div>`;
    }).join("");
    timeline.innerHTML = `<p class="timeline-title">화자 타임라인</p><div class="timeline-scale"><span>00:00</span><span>${duration(durationValue)}</span></div>${rows}`;
  }
  function stopWatching() { if (events) { events.close(); events = null; } if (polling) { clearInterval(polling); polling = null; } }
  function watch(id) {
    stopWatching();
    events = new EventSource(apiUrl(`/api/jobs/${id}/events`, true));
    events.addEventListener("status", (event) => render(JSON.parse(event.data)));
    events.onerror = () => { events.close(); events = null; polling = setInterval(() => request(`/api/jobs/${id}`).then(render).catch(() => stopWatching()), 2000); };
  }
  function updateProfileState() {
    const selected = profiles.find((profile) => profile.key === $("profile").value);
    const status = $("profile-status"), button = $("submit"), form = $("upload-form"), setup = $("server-setup");
    const available = profiles.filter((profile) => profile.available);
    if (!available.length) {
      const preferred = profiles.find((profile) => profile.key === "turbo") || profiles[0];
      form.classList.add("hidden");
      setup.classList.remove("hidden");
      setup.textContent = `이 서버에는 지금 실행할 전사 모델이 없어 분석 요청을 받을 수 없습니다. ${preferred?.unavailable_reason || "모델을 준비한 뒤 페이지를 새로고침해 주세요."}`;
      return;
    }
    form.classList.remove("hidden");
    setup.classList.add("hidden");
    if (!selected) { status.textContent = "준비된 전사 모델을 선택해 주세요."; status.className = "hint error"; button.disabled = true; return; }
    status.textContent = selected.available ? selected.description : `준비 필요: ${selected.unavailable_reason}`;
    status.className = selected.available ? "hint" : "hint error";
    button.disabled = !selected.available;
  }
  function setLlmStatus(message, isError = false) { const status = $("llm-status"); status.textContent = message; status.classList.toggle("error", isError); }
  function llmInput(includeModel = false) {
    const input = {base_url: $("llm-url").value.trim(), api_key: $("llm-api-key").value};
    if (includeModel) { input.model = $("llm-model").value.trim(); if (!input.model) throw new Error("연결을 확인할 모델을 선택하거나 직접 입력해 주세요."); }
    return input;
  }
  function setLlmBusy(value) { $("llm-load-models").disabled = value; $("llm-verify").disabled = value; }
  async function loadLlmConnection() {
    try {
      const connection = await request("/api/local-llm");
      if (connection.base_url) $("llm-url").value = connection.base_url;
      else if (!$("llm-url").value) $("llm-url").value = "http://127.0.0.1:8000";
      if (connection.model) $("llm-model").value = connection.model;
      setLlmStatus(connection.message, false);
    } catch (error) { setLlmStatus(error.message, true); }
  }
  async function loadLlmModels() {
    setLlmBusy(true); setLlmStatus("모델 목록을 읽고 있습니다…");
    try {
      const data = await request("/api/local-llm/models", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(llmInput())});
      const ids = data.models.map((model) => model.id);
      $("llm-model-options").innerHTML = ids.map((id) => `<option value="${escapeHtml(id)}">`).join("");
      if (ids.length && !ids.includes($("llm-model").value.trim())) $("llm-model").value = ids[0];
      setLlmStatus(ids.length ? `모델 ${ids.length}개를 불러왔습니다. 사용할 모델을 고른 뒤 연결 확인을 눌러 주세요.` : "서버가 모델 목록을 비워서 반환했습니다. 모델 ID를 직접 입력해 주세요.");
    } catch (error) { setLlmStatus(error.message, true); } finally { setLlmBusy(false); }
  }
  async function verifyLlmConnection() {
    let input;
    try { input = llmInput(true); } catch (error) { setLlmStatus(error.message, true); return; }
    setLlmBusy(true); setLlmStatus("선택한 모델에 짧은 연결 요청을 보내고 있습니다…");
    try {
      const connection = await request("/api/local-llm/verify", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(input)});
      const suffix = connection.response_preview ? ` 응답: ${connection.response_preview}` : "";
      setLlmStatus(`연결되었습니다. 선택한 모델이 짧은 요청을 처리했습니다.${suffix}`);
      $("llm-api-key").value = "";
    } catch (error) { setLlmStatus(error.message, true); } finally { setLlmBusy(false); }
  }
  async function loadProfiles() {
    const data = await request("/api/profiles"); profiles = data.profiles;
    const select = $("profile");
    select.innerHTML = profiles.map((profile) => `<option value="${escapeHtml(profile.key)}"${profile.available ? "" : " disabled"}>${escapeHtml(profile.label)}${profile.available ? "" : " · 준비 필요"}</option>`).join("");
    const preferred = profiles.find((profile) => profile.key === "turbo" && profile.available) || profiles.find((profile) => profile.available);
    if (preferred) select.value = preferred.key;
    select.disabled = !preferred;
    updateProfileState();
  }
  async function refreshProtectedData() {
    $("upload-status").textContent = "서버 연결을 확인하고 있습니다…";
    try { await Promise.all([loadProfiles(), loadJobs()]); $("upload-status").textContent = "서버에 연결했습니다."; }
    catch (error) { $("upload-status").textContent = error.message; }
  }
  async function loadJobs() { try { const data = await request("/api/jobs"); $("jobs").innerHTML = data.jobs.length ? data.jobs.slice(0,12).map((job) => `<div class="job"><strong>${escapeHtml(job.source_name)}</strong><br><span class="hint">${escapeHtml(job.state)} · ${job.progress}% · ${escapeHtml(job.message)}</span></div>`).join("") : "아직 요청한 작업이 없습니다."; } catch (error) { $("jobs").textContent = error.message; } }
  $("profile").addEventListener("change", updateProfileState);
  $("llm-load-models")?.addEventListener("click", loadLlmModels);
  $("llm-verify")?.addEventListener("click", verifyLlmConnection);
  $("token")?.addEventListener("change", refreshProtectedData);
  $("upload-form").addEventListener("submit", async (event) => { event.preventDefault(); const selected = profiles.find((profile) => profile.key === $("profile").value); if (!selected || !selected.available) { updateProfileState(); return; } const form = new FormData(event.currentTarget); const button = $("submit"); button.disabled = true; $("upload-status").textContent = "파일을 서버로 보내고 있습니다…"; try { const job = (await request("/api/jobs", {method:"POST", body:form})).job; render(job); watch(job.id); loadJobs(); $("upload-status").textContent = "분석 대기열에 추가했습니다."; } catch (error) { $("upload-status").textContent = error.message; } finally { updateProfileState(); } });
  $("cancel").addEventListener("click", async () => { if (!activeId) return; try { render((await request(`/api/jobs/${activeId}/cancel`, {method:"POST"})).job); } catch (error) { $("current-notice").textContent = error.message; $("current-notice").classList.add("error"); } });
  Promise.all([loadProfiles(), loadJobs(), $("local-ai") ? loadLlmConnection() : Promise.resolve()]).catch((error) => { $("upload-status").textContent = error.message; });
</script>
</body></html>"""
    local_llm_section = """
  <section id="local-ai">
    <h2>로컬 AI 연결</h2>
    <p class="hint">이 서버 PC에서 실행 중인 OpenAI 호환 로컬 AI 서버에만 연결합니다. 서버 주소와 모델만 저장하며 API 키는 이번 연결 확인에만 사용합니다.</p>
    <div class="grid"><div><label for="llm-url">서버 주소</label><input id="llm-url" type="text" inputmode="url" placeholder="http://127.0.0.1:8000"></div>
    <div><label for="llm-api-key">API 키 (선택)</label><input id="llm-api-key" type="password" autocomplete="off" placeholder="필요한 서버에서만 입력"></div></div>
    <div class="grid"><div><label for="llm-model">모델</label><input id="llm-model" type="text" list="llm-model-options" placeholder="모델 불러오기 또는 직접 입력"><datalist id="llm-model-options"></datalist></div>
    <div class="actions"><button id="llm-load-models" class="secondary" type="button">모델 불러오기</button><button id="llm-verify" type="button">연결 확인</button></div></div>
    <p id="llm-status" class="notice">로컬 AI 연결 상태를 확인하고 있습니다…</p>
  </section>
    """ if local_llm_available else ""
    token_section = """
      <label for="token">접근 토큰</label><input id="token" type="password" autocomplete="off" placeholder="서버에 설정한 토큰을 입력한 뒤 다른 곳을 클릭하세요">
      <p class="hint">이 서버는 로컬 네트워크에 열려 있어 접근 토큰이 필요합니다.</p>
    """ if access_token_required else ""
    return page.replace("__LOCAL_LLM_SECTION__", local_llm_section).replace("__ACCESS_TOKEN_SECTION__", token_section)


def create_app(
    *,
    project_root: Path = PROJECT_ROOT,
    access_token: str | None = None,
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    manager: JobManager | None = None,
    llm_connection_manager: LocalLLMConnectionManager | None = None,
    local_llm_enabled: bool = True,
) -> FastAPI:
    """Create a server application without importing the Tkinter GUI."""

    if max_upload_mb < 1:
        raise ValueError("최대 업로드 크기는 1MB 이상이어야 합니다.")
    if retention_days < 1:
        raise ValueError("결과 보관 기간은 하루 이상이어야 합니다.")
    token = (access_token or "").strip() or None
    job_manager = manager or JobManager(project_root)
    local_llm_manager = llm_connection_manager or LocalLLMConnectionManager()
    job_manager.cleanup_expired(retention_days)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        job_manager.shutdown()

    app = FastAPI(title="TalkTracker Server", version="0.1.0", lifespan=lifespan)
    app.state.job_manager = job_manager
    app.state.local_llm_manager = local_llm_manager
    app.state.local_llm_enabled = local_llm_enabled
    app.state.access_token_required = bool(token)
    app.state.max_upload_bytes = max_upload_mb * 1024 * 1024

    def require_access(
        x_talktracker_token: Annotated[str | None, Header()] = None,
        access_token_query: Annotated[str | None, Query(alias="access_token")] = None,
    ) -> None:
        if not token:
            return
        candidate = x_talktracker_token or access_token_query or ""
        if not hmac.compare_digest(candidate, token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="접근 토큰이 필요합니다.")

    def require_local_llm() -> None:
        if not local_llm_enabled:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="로컬 AI 연결은 이 PC에서 연 웹 화면에서만 사용할 수 있습니다.")

    def snapshot_or_404(job_id: str) -> dict[str, Any]:
        try:
            return job_manager.snapshot(job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="작업을 찾지 못했습니다.") from error

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        profiles = _profile_payload(project_root)
        if not any(profile["available"] for profile in profiles):
            return _server_unavailable_page()
        return _index_page(access_token_required=bool(token), local_llm_available=local_llm_enabled)

    @app.get("/api/health")
    def health(_: None = Depends(require_access)) -> dict[str, Any]:
        cuda = get_cuda_runtime_status()
        jobs = job_manager.list_snapshots()
        return {
            "status": "ok",
            "cuda": {"available": cuda.available, "message": cuda.message},
            "queue": dict(Counter(job["state"] for job in jobs)),
            "access_token_required": bool(token),
            "max_upload_mb": max_upload_mb,
            "retention_days": retention_days,
        }

    @app.get("/api/profiles")
    def profiles(_: None = Depends(require_access)) -> dict[str, Any]:
        return {"profiles": _profile_payload(project_root)}

    @app.get("/api/local-llm")
    def local_llm_status(_: None = Depends(require_access), _local: None = Depends(require_local_llm)) -> dict[str, Any]:
        connection = local_llm_manager.snapshot()
        return {
            "configured": connection.settings.is_configured,
            "connected": connection.connected,
            "message": connection.message,
            "base_url": connection.settings.base_url,
            "model": connection.settings.model,
        }

    @app.post("/api/local-llm/models")
    async def local_llm_models(
        request: Request,
        _: None = Depends(require_access),
        _local: None = Depends(require_local_llm),
    ) -> dict[str, Any]:
        base_url, api_key, _model = await _read_local_llm_request(request, require_model=False)
        try:
            models = local_llm_manager.list_models(base_url, api_key)
        except LocalLLMError as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
        return {"models": [{"id": model.id, "owned_by": model.owned_by} for model in models]}

    @app.post("/api/local-llm/verify")
    async def verify_local_llm(
        request: Request,
        _: None = Depends(require_access),
        _local: None = Depends(require_local_llm),
    ) -> dict[str, Any]:
        base_url, api_key, model = await _read_local_llm_request(request, require_model=True)
        try:
            result = local_llm_manager.verify(base_url, api_key, model)
        except LocalLLMError as error:
            local_llm_manager.mark_disconnected()
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
        connection = local_llm_manager.snapshot()
        return {
            "configured": connection.settings.is_configured,
            "connected": connection.connected,
            "message": connection.message,
            "base_url": connection.settings.base_url,
            "model": result.model,
            "response_preview": result.response_preview,
        }

    @app.get("/api/jobs")
    def jobs(_: None = Depends(require_access)) -> dict[str, Any]:
        return {"jobs": job_manager.list_snapshots()}

    @app.post("/api/jobs", status_code=status.HTTP_201_CREATED)
    async def create_job(
        file: Annotated[UploadFile, File(...)],
        profile: Annotated[str, Form()] = "turbo",
        _: None = Depends(require_access),
    ) -> JSONResponse:
        try:
            profile_definition = next(item for item in WHISPER_PROFILES if item.key == profile)
        except StopIteration:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="알 수 없는 Whisper 프로필입니다.") from None
        available, reason = _profile_availability(project_root, profile_definition)
        if not available:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=reason or "선택한 모델을 실행할 수 없습니다.")
        try:
            job = job_manager.create_upload_job(file.filename or "recording", profile)
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

        total = 0
        try:
            with job.source_path.open("wb") as destination:
                while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                    total += len(chunk)
                    if total > app.state.max_upload_bytes:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"업로드 파일은 최대 {max_upload_mb}MB까지 지원합니다.",
                        )
                    destination.write(chunk)
        except HTTPException as error:
            job.source_path.unlink(missing_ok=True)
            job_manager.fail_upload(job.id, error.detail)
            raise
        except Exception as error:
            job.source_path.unlink(missing_ok=True)
            job_manager.fail_upload(job.id, "파일 업로드 중 오류가 발생했습니다.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="파일 업로드 중 오류가 발생했습니다.") from error
        finally:
            await file.close()

        return JSONResponse({"job": job_manager.finish_upload(job.id, total)}, status_code=status.HTTP_201_CREATED)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, _: None = Depends(require_access)) -> dict[str, Any]:
        return {"job": snapshot_or_404(job_id)}

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(
        request: Request,
        job_id: str,
        _: None = Depends(require_access),
    ) -> StreamingResponse:
        snapshot_or_404(job_id)

        async def stream() -> AsyncIterator[str]:
            revision = -1
            while not await request.is_disconnected():
                try:
                    job = job_manager.snapshot(job_id)
                except JobNotFoundError:
                    yield "event: deleted\ndata: {}\n\n"
                    return
                if job["revision"] != revision:
                    revision = job["revision"]
                    yield f"id: {revision}\nevent: status\ndata: {json.dumps(job, ensure_ascii=False)}\n\n"
                if job["state"] in {"completed", "cancelled", "failed"}:
                    return
                await asyncio.sleep(0.75)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, _: None = Depends(require_access)) -> dict[str, Any]:
        try:
            return {"job": job_manager.cancel(job_id)}
        except JobNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="작업을 찾지 못했습니다.") from error

    @app.get("/api/jobs/{job_id}/result.md")
    def download_result(job_id: str, _: None = Depends(require_access)) -> FileResponse:
        try:
            result_path = job_manager.result_path(job_id)
            job = job_manager.snapshot(job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="작업을 찾지 못했습니다.") from error
        except JobStateError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        filename = f"{Path(job['source_name']).stem}-talktracker.md"
        return FileResponse(result_path, media_type="text/markdown; charset=utf-8", filename=filename)

    @app.delete("/api/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_job(job_id: str, _: None = Depends(require_access)) -> None:
        try:
            job_manager.delete(job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="작업을 찾지 못했습니다.") from error
        except JobStateError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

    return app


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TalkTracker local web server")
    parser.add_argument("--host", default=os.environ.get("TALKTRACKER_WEB_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=_integer_setting("TALKTRACKER_WEB_PORT", DEFAULT_PORT, minimum=1))
    parser.add_argument(
        "--max-upload-mb",
        type=int,
        default=_integer_setting("TALKTRACKER_WEB_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB, minimum=1),
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=_integer_setting("TALKTRACKER_WEB_RETENTION_DAYS", DEFAULT_RETENTION_DAYS, minimum=1),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    access_token = os.environ.get("TALKTRACKER_WEB_TOKEN", "").strip() or None
    if not _is_loopback_host(arguments.host) and not access_token:
        print(
            "Warning: the server is open to the local network without an access token. "
            "Use only on a trusted internal network; never expose this address to the internet.",
            flush=True,
        )

    import uvicorn

    app = create_app(
        project_root=PROJECT_ROOT,
        access_token=access_token,
        max_upload_mb=arguments.max_upload_mb,
        retention_days=arguments.retention_days,
        local_llm_enabled=_is_loopback_host(arguments.host),
    )
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="info")


if __name__ == "__main__":
    main()
