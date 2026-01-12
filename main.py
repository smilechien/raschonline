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

from flask import Flask, request, jsonify, send_from_directory, abort

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
