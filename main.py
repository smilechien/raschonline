# -*- coding: utf-8 -*-
"""
main.py — Flask runner for raschonline.py
- Auto-open HTML report after successful run (local only)
"""

from __future__ import annotations
import os
# --- JSON strict helpers (NaN/Inf -> null) ---
def _json_sanitize(x):
    try:
        import numpy as _np
    except Exception:
        _np = None

    if x is None:
        return None

    if _np is not None and isinstance(x, (_np.floating, _np.integer, _np.bool_)):
        x = x.item()

    if isinstance(x, float):
        if (x != x) or x == float("inf") or x == float("-inf"):
            return None
        return x

    if isinstance(x, (int, bool, str)):
        return x

    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]

    if _np is not None and isinstance(x, _np.ndarray):
        return [_json_sanitize(v) for v in x.tolist()]

    # fallback: try object dict, else stringify
    try:
        return _json_sanitize(x.__dict__)
    except Exception:
        return str(x)

def _safe_json_dumps(obj, **kwargs):
    import json as _json
    kwargs.setdefault("allow_nan", False)
    return _json.dumps(_json_sanitize(obj), **kwargs)

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import raschonline  # local module (raschonline.py next to this file)
except Exception as e:
    raise RuntimeError(f"raschonline import failed: {e!r}") from e

import shutil
import json
import traceback
import threading
import webbrowser
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, send_from_directory, abort, redirect

# ---------------------------------------------------------------------
# URL query runner (GAE-friendly)
#   /?example=1                    -> run bundled demo.csv
#   /?url=https://.../file.csv      -> fetch CSV and run
# Security:
#   - allowlist (env URL_FETCH_ALLOWLIST, comma-separated hostnames)
#   - max size (env URL_FETCH_MAX_BYTES, default 10MB)
#   - timeout (env URL_FETCH_TIMEOUT, default 12s)
#   - block localhost/private/link-local/reserved IPs & metadata hosts
# ---------------------------------------------------------------------
import urllib.parse
import urllib.request
import socket
import ipaddress

def _parse_allowlist(default_host: str) -> set[str]:
    raw = (os.environ.get("URL_FETCH_ALLOWLIST") or "").strip()
    if raw:
        items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    else:
        # sensible defaults; always allow current host so ?url can point to /demo.csv
        items = [default_host.lower(), "raw.githubusercontent.com", "githubusercontent.com", "storage.googleapis.com"]
    return set(items)


def _host_allowed(host: str, allow: set[str]) -> bool:
    h = (host or "").lower().strip(".")
    if not h:
        return False
    # exact match or subdomain of an allowlisted domain
    return any(h == a or h.endswith("." + a) for a in allow)


def _resolve_public_ips(host: str) -> list[str]:
    """Resolve host -> IPs; raise if any resolved IP is non-public."""
    bad_hosts = {"metadata.google.internal", "metadata"}
    if host.lower() in bad_hosts:
        raise ValueError("Blocked metadata host")

    ips: list[str] = []
    for family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
        ip = sockaddr[0]
        ips.append(ip)
        ip_obj = ipaddress.ip_address(ip)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
        ):
            raise ValueError(f"Blocked non-public IP: {ip}")
        # Explicitly block common metadata IP
        if str(ip_obj) == "169.254.169.254":
            raise ValueError("Blocked metadata IP")
    # If resolution returns nothing, treat as blocked
    if not ips:
        raise ValueError("Host resolution failed")
    return ips


def _fetch_url_to_tmp(url: str, run_id: str, default_host: str) -> Path:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs are allowed")
    if not parsed.netloc:
        raise ValueError("URL must include a hostname")

    host = parsed.hostname or ""
    allow = _parse_allowlist(default_host)
    if not _host_allowed(host, allow):
        raise ValueError(f"Host not in allowlist: {host}")

    # SSRF guard: resolve and ensure public IPs
    _resolve_public_ips(host)

    max_bytes = int(os.environ.get("URL_FETCH_MAX_BYTES", str(10 * 1024 * 1024)))
    timeout = float(os.environ.get("URL_FETCH_TIMEOUT", "12"))

    req = urllib.request.Request(url, headers={"User-Agent": "RaschOnline/1.0"})
    out_path = (REPORTS_DIR / run_id / "url.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ct = (resp.headers.get("Content-Type") or "").lower()
        # allow text/csv or text/plain; don't hard-fail if missing
        if ct and ("text" not in ct and "csv" not in ct and "octet-stream" not in ct):
            # still allow; many servers mislabel CSV
            pass
        with open(out_path, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"File too large (>{max_bytes} bytes)")
                f.write(chunk)

    if total == 0:
        raise ValueError("Downloaded file was empty")
    return out_path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

# ✅ GAE Standard: the application directory (/workspace) is read-only.
# Use /tmp for generated reports (ephemeral storage).
import tempfile
_tmp_base = Path(tempfile.gettempdir())

# Allow override via env var; defaults to /tmp/reports on GAE (or OS temp dir locally)
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", str(_tmp_base / "reports")))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Import Rasch engine (must be local raschonline.py)
# ---------------------------------------------------------------------
try:
    _IMPORT_ERROR = None
except Exception as e:
    raschonline = None
    _IMPORT_ERROR = repr(e)

# ---------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _is_cloud_runtime() -> bool:
    for k in ("GAE_ENV", "K_SERVICE", "K_REVISION", "GOOGLE_CLOUD_PROJECT"):
        if k in os.environ:
            return True
    return False


def _auto_open_report(report_rel_url: str):
    """Open report.html in browser (local only)."""
    try:
        if _is_cloud_runtime():
            return
        port = int(os.environ.get("PORT", "8000"))
        url = f"http://127.0.0.1:{port}{report_rel_url}"
        threading.Thread(
            target=lambda: webbrowser.open_new_tab(url),
            daemon=True
        ).start()
    except Exception:
        pass


def _new_run_id() -> str:
    import secrets
    return secrets.token_hex(6)

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
def home():
    # -------------------------------------------------------------
    # Workflow-friendly query parameters
    #   /?example=1
    #   /?url=https://.../data.csv
    # Optional extra params (passed into engine): mode, max_category,
    # continuous_transform, kid_no, item_no, ...
    # -------------------------------------------------------------
    try:
        example = (request.args.get("example") or "").strip()
        url = (request.args.get("url") or "").strip()

        if example == "1" or url:
            run_id = _new_run_id()
            # allowlist defaults include current host
            default_host = (request.host.split(":", 1)[0] if request.host else "").strip()

            if example == "1":
                demo = STATIC_DIR / "demo.csv"
                if not demo.exists():
                    return "Missing static/demo.csv", 404
                csv_path = demo
            else:
                csv_path = _fetch_url_to_tmp(url, run_id=run_id, default_host=default_host)

            # Pass other query params into the engine (but not url/example)
            form = {k: v for k, v in request.args.items() if k not in ("url", "example")}
            payload = _run_engine(Path(csv_path), run_id, form)
            rep = payload.get("report_url") or payload.get("report_html_url")
            if rep:
                return redirect(rep)
            return jsonify(_json_sanitize(payload))

    except Exception as e:
        # Return a readable message instead of a blank 500 page.
        return (
            jsonify({
                "ok": False,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }),
            500,
        )

    # Default: serve the UI
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        return "Missing static/index.html", 404
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "raschonline_import_ok": raschonline is not None,
        "import_error": _IMPORT_ERROR
    })


@app.get("/demo.csv")
def demo_csv():
    demo = STATIC_DIR / "demo.csv"
    if not demo.exists():
        return "Missing static/demo.csv", 404
    return send_from_directory(STATIC_DIR, "demo.csv", as_attachment=True)


# ---------------------------------------------------------------------
# Core engine runner
# ---------------------------------------------------------------------
def _run_engine(csv_path: Path, run_id: str, form: Dict[str, str]) -> Dict[str, Any]:
    if raschonline is None:
        raise RuntimeError(f"raschonline import failed: {_IMPORT_ERROR}")

    run_dir = REPORTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_copy = run_dir / "input.csv"
    input_copy.write_bytes(csv_path.read_bytes())

    mode = (form.get("mode") or "auto").strip().lower()
    model = (form.get("model") or "").strip().lower()
    if (not mode or mode == "auto") and model:
        mode = model

    continuous_transform = (form.get("continuous_transform") or "linear").strip().lower()

    max_category: Optional[int] = None
    if form.get("max_category"):
        max_category = int(form["max_category"])


    # KIDMAP target (1-based ENTRY)
    try:
        kid_no = int((form.get("kid_no") or "1").strip())
    except Exception:
        kid_no = 1
    if kid_no < 1:
        kid_no = 1

    # ICC/CPC target item (1-based item index among response columns)
    try:
        item_no = int((form.get("item_no") or "1").strip())
    except Exception:
        item_no = 1
    if item_no < 1:
        item_no = 1



    # ------------------ Run Rasch ------------------
    engine = (form.get("engine") or "python").strip().lower()
    engine = "python"

    if False:
        pass
    else:
        # Pure-Python engine (fallback)
        res = raschonline.run_rasch(
            str(input_copy),
            mode=mode,
            continuous_transform=continuous_transform,
            max_category=max_category,
            max_iter=100,
        )

    # ------------------ Save outputs ------------------
    res.person_df.to_csv(run_dir / "person_estimates.csv", index=False, encoding="utf-8-sig")
    res.item_df.to_csv(run_dir / "item_estimates.csv", index=False, encoding="utf-8-sig")
    # Avoid SameFileError when input_copy already points to run_dir / 'input.csv'
    _src = os.path.abspath(str(input_copy))
    _dst = os.path.abspath(str(run_dir / "input.csv"))
    if _src != _dst:
        shutil.copyfile(_src, _dst)

    payload: Dict[str, Any] = {
        "ok": bool(res.ok),
        "run_id": run_id,
        "model": res.model,
        "iterations": res.iterations,
        "last_error": res.last_error,
        "stop_reason": res.stop_reason,
        "min_cat": res.min_cat,
        "max_cat": res.max_cat,
        "report_url": f"/reports/{run_id}/report.html",
        "report_html_url": f"/reports/{run_id}/report.html",

        "person_csv_url": f"/reports/{run_id}/person_estimates.csv",
        "item_csv_url": f"/reports/{run_id}/item_estimates.csv",
        "input_csv_url": f"/reports/{run_id}/input.csv",
        "xfile_csv_url": f"/reports/{run_id}/tam/xfile.csv",
        "residual_person_csv_url": f"/reports/{run_id}/tam/residual_person.csv",
        "residual_item_csv_url": f"/reports/{run_id}/tam/residual_item.csv",
    }

    # Render report
    try:
        html = raschonline.render_report_html(res, run_id, kid_no=kid_no, item_no=item_no)
    except Exception as e:
        html = f"<html><body><h1>Report failed</h1><pre>{e!r}</pre></body></html>"

    (run_dir / "report.html").write_text(html, encoding="utf-8")

    # ---- DEBUG: ENGINE payload preview (first 30 lines) ----
    try:
        _preview = _safe_json_dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        print("\n[DEBUG] ENGINE payload preview (first 30 lines)")
        for _i, _line in enumerate(_preview.splitlines()[:30], 1):
            print(f"{_i:02d}: {_line}")
    except Exception as _e:
        print("[DEBUG] failed to preview engine payload:", repr(_e))
    # -------------------------------------------------------
    (run_dir / "result.json").write_text(
        _safe_json_dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8"
    )

    return payload

# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------
@app.get("/run_demo")
def run_demo_get():
    return "Use POST /run_demo (via button or curl).", 200


@app.post("/run_demo")
def run_demo():
    try:
        demo = STATIC_DIR / "demo.csv"
        if not demo.exists():
            return jsonify({"ok": False, "error": "Missing static/demo.csv"}), 404

        run_id = _new_run_id()
        payload = _run_engine(demo, run_id, dict(request.form))

        # auto-open
        u = payload.get("report_url") or payload.get("report_html_url")
        if u:
            _auto_open_report(u)


        # ---- DEBUG: print first 30 lines of JSON payload (like DevTools Network → Response preview) ----

        try:

            _preview = _safe_json_dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)

            print("\n[DEBUG] JSON response preview (first 30 lines)")

            for _i, _line in enumerate(_preview.splitlines()[:30], 1):

                print(f"{_i:02d}: {_line}")

        except Exception as _e:

            print("[DEBUG] failed to preview payload:", repr(_e))

        # -------------------------------------------------------------------------------

        return jsonify(_json_sanitize(payload))
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": repr(e),
            "traceback": traceback.format_exc()
        }), 500

# ---------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------
@app.get("/run")
def run_get():
    return "Use POST /run with multipart/form-data field 'file'.", 200


@app.post("/run")
def run_upload():
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No file field"}), 400

        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"ok": False, "error": "Empty upload"}), 400

        run_id = _new_run_id()
        run_dir = REPORTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        up_path = run_dir / "upload.csv"
        f.save(str(up_path))

        payload = _run_engine(up_path, run_id, dict(request.form))

        # ---- DEBUG: print first 30 lines of JSON payload (like DevTools Network → Response preview) ----
        try:
            _preview = _safe_json_dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
            print("\n[DEBUG] JSON response preview (first 30 lines)")
            for _i, _line in enumerate(_preview.splitlines()[:30], 1):
                print(f"{_i:02d}: {_line}")
        except Exception as _e:
            print("[DEBUG] failed to preview payload:", repr(_e))
        # -------------------------------------------------------------------------------
        return jsonify(_json_sanitize(payload))
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": repr(e),
            "traceback": traceback.format_exc()
        }), 500

# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------
@app.get("/reports/<run_id>/<path:filename>")
def get_report_file(run_id: str, filename: str):
    run_dir = REPORTS_DIR / run_id
    if not run_dir.exists():
        abort(404)
    return send_from_directory(run_dir, filename)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        debug=True,
        use_reloader=False,
    )
