"""Minimal local server: static site + POST /api/video-score.

Usage:
  python -m cello_note_detector.web [--port 8765] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import shutil
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from cello_note_detector.web.pipeline import VIDEO_EXTS, SilentAudioError, process_video_to_score

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STATIC_ROOT = _REPO_ROOT
_UPLOAD_ROOT = _REPO_ROOT / "outputs" / "web_jobs"
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _json_bytes(payload: dict[str, Any], status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(payload, indent=2).encode("utf-8")
    return status, body, "application/json; charset=utf-8"


def _safe_job_dir(job_id: str) -> Path:
    if not job_id or any(c in job_id for c in "/\\.."):
        raise ValueError("invalid job id")
    path = (_UPLOAD_ROOT / job_id).resolve()
    if not str(path).startswith(str(_UPLOAD_ROOT.resolve())):
        raise ValueError("invalid job id")
    return path


def _rel_artifact(path: str | None, job_id: str) -> str | None:
    if not path:
        return None
    p = Path(path).resolve()
    job_dir = _safe_job_dir(job_id)
    try:
        rel = p.relative_to(job_dir)
    except ValueError:
        return None
    return f"/api/jobs/{job_id}/files/{rel.as_posix()}"


def _public_result(job_id: str, result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    out["job_id"] = job_id
    out["musicxml_url"] = _rel_artifact(result.get("musicxml_path"), job_id)
    out["score_json_url"] = _rel_artifact(result.get("score_json_path"), job_id)
    out["fused_json_url"] = _rel_artifact(result.get("fused_json_path"), job_id)
    out["overlay_urls"] = [
        u
        for u in (_rel_artifact(p, job_id) for p in result.get("overlay_previews") or [])
        if u
    ]
    # Drop absolute machine paths from the browser payload.
    for key in (
        "musicxml_path",
        "score_json_path",
        "fused_json_path",
        "sample_json_path",
        "audio_path",
        "overlay_previews",
    ):
        out.pop(key, None)
    return out


def _run_job(job_id: str, video_path: Path, options: dict[str, Any]) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id]["status"] = "running"
        _JOBS[job_id]["message"] = "Running AV fuse + MusicXML…"
        _JOBS[job_id]["started_at"] = time.time()
    try:
        result = process_video_to_score(
            video_path,
            _safe_job_dir(job_id),
            max_frames=options.get("max_frames"),
            sample_every=int(options.get("sample_every") or 2),
            overlay=bool(options.get("overlay", True)),
            quantize=str(options.get("quantize") or "coarse"),
            write_video_only=False,
        )
        public = _public_result(job_id, result)
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "done"
            _JOBS[job_id]["message"] = (
                f"Done — {result.get('num_notes', 0)} notes"
                f" ({result.get('key') or '?'}, "
                f"{result.get('tempo_bpm') or '?'} BPM)"
            )
            _JOBS[job_id]["result"] = public
            _JOBS[job_id]["finished_at"] = time.time()
    except SilentAudioError as exc:
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["message"] = str(exc)
            _JOBS[job_id]["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 — surface to UI
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["message"] = f"Pipeline failed: {exc}"
            _JOBS[job_id]["error"] = str(exc)
            _JOBS[job_id]["traceback"] = traceback.format_exc()[-2000:]


class Handler(BaseHTTPRequestHandler):
    server_version = "cello-note-detector-web/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        code, body, ctype = _json_bytes(payload, status)
        self._send(code, body, ctype)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/health":
            self._send_json({"ok": True, "service": "cello-note-detector-web"})
            return

        if path.startswith("/api/jobs/") and path.endswith("/status"):
            parts = path.strip("/").split("/")
            # api/jobs/<id>/status
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "jobs":
                job_id = parts[2]
                with _JOBS_LOCK:
                    job = _JOBS.get(job_id)
                if not job:
                    self._send_json({"ok": False, "error": "unknown job"}, 404)
                    return
                payload = {
                    "ok": True,
                    "job_id": job_id,
                    "status": job["status"],
                    "message": job.get("message"),
                    "error": job.get("error"),
                    "result": job.get("result"),
                }
                self._send_json(payload)
                return

        if path.startswith("/api/jobs/") and "/files/" in path:
            # /api/jobs/<id>/files/<rel>
            rest = path[len("/api/jobs/") :]
            job_id, _, rel = rest.partition("/files/")
            try:
                job_dir = _safe_job_dir(job_id)
                file_path = (job_dir / rel).resolve()
                if not str(file_path).startswith(str(job_dir)):
                    raise ValueError("path escape")
                if not file_path.is_file():
                    self._send_json({"ok": False, "error": "file not found"}, 404)
                    return
            except ValueError:
                self._send_json({"ok": False, "error": "invalid path"}, 400)
                return
            ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header(
                "Content-Disposition",
                f'inline; filename="{file_path.name}"',
            )
            self.end_headers()
            self.wfile.write(data)
            return

        # Static files from repo root.
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        if ".." in rel.split("/"):
            self._send_json({"ok": False, "error": "not found"}, 404)
            return
        file_path = (_STATIC_ROOT / rel).resolve()
        if not str(file_path).startswith(str(_STATIC_ROOT.resolve())) or not file_path.is_file():
            self._send_json({"ok": False, "error": "not found"}, 404)
            return
        ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self._send(200, data, ctype)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/video-score":
            self._send_json({"ok": False, "error": "not found"}, 404)
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json(
                {"ok": False, "error": "expected multipart/form-data with a video file"},
                400,
            )
            return

        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
        }
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ=environ,
            keep_blank_values=True,
        )
        file_item = form["file"] if "file" in form else None
        if file_item is None or not getattr(file_item, "filename", None):
            self._send_json({"ok": False, "error": "missing file field"}, 400)
            return

        filename = Path(str(file_item.filename)).name
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_EXTS:
            self._send_json(
                {
                    "ok": False,
                    "error": f"unsupported type {suffix!r}; use mp4/webm/mov",
                },
                400,
            )
            return

        max_frames_raw = form.getvalue("max_frames")
        max_frames: int | None = None
        if max_frames_raw not in (None, "", "0", "none", "full"):
            try:
                max_frames = int(max_frames_raw)
            except ValueError:
                self._send_json({"ok": False, "error": "max_frames must be an int"}, 400)
                return

        quantize = str(form.getvalue("quantize") or "coarse")
        sample_every = int(form.getvalue("sample_every") or 2)
        overlay = str(form.getvalue("overlay") or "1") not in ("0", "false", "False")

        job_id = uuid.uuid4().hex[:12]
        job_dir = _UPLOAD_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        # Keep a readable stem from the original filename (sanitized).
        stem = Path(filename).stem
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:80] or "upload"
        video_path = job_dir / f"{safe_stem}{suffix}"
        with open(video_path, "wb") as fh:
            shutil.copyfileobj(file_item.file, fh)

        with _JOBS_LOCK:
            _JOBS[job_id] = {
                "status": "queued",
                "message": "Queued…",
                "created_at": time.time(),
                "filename": filename,
            }

        options = {
            "max_frames": max_frames,
            "quantize": quantize,
            "sample_every": sample_every,
            "overlay": overlay,
        }
        thread = threading.Thread(
            target=_run_job,
            args=(job_id, video_path, options),
            daemon=True,
        )
        thread.start()
        self._send_json(
            {
                "ok": True,
                "job_id": job_id,
                "status_url": f"/api/jobs/{job_id}/status",
                "message": "Job started — poll status_url",
            },
            202,
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Cello video → MusicXML local website")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--clean-jobs",
        action="store_true",
        help="Delete outputs/web_jobs before starting",
    )
    args = p.parse_args(argv)
    if args.clean_jobs and _UPLOAD_ROOT.exists():
        shutil.rmtree(_UPLOAD_ROOT)
    _UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Soli web UI → http://{args.host}:{args.port}/")
    print("Upload a cello video (mp4/webm/mov) → fused MusicXML with fingerings/positions.")
    print(f"Demo clip: {_REPO_ROOT / 'data/video/clips/mspho_a_major_3oct_front.mp4'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
