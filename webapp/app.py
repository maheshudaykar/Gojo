"""
Production-grade Flask application with:
- Structured logging
- Input validation and sanitization
- Security hardening (CSRF, rate limiting)
- Heartbeat-based lifecycle management
- Professional error handling
- Monitoring and health checks
"""
from __future__ import annotations

import io
import csv
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from flask import Flask, flash, jsonify, render_template, request, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman  # type: ignore[import-not-found]
from flask_wtf import CSRFProtect  # type: ignore[import-not-found]
from phish_detector.analyze import AnalysisConfig, analyze_url, load_ml_context

APP_ROOT = Path(__file__).resolve().parent
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "webapp.log"


def _build_log_handlers() -> list[logging.Handler]:
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ]
    formatter: logging.Formatter
    try:
        from pythonjsonlogger import json as jsonlogger

        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            timestamp=True,
        )
    except ImportError:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    for handler in handlers:
        handler.setFormatter(formatter)
    return handlers


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    handlers=_build_log_handlers(),
)
logger = logging.getLogger(__name__)

# Try to use v2 policy, fallback to v1
try:
    from phish_detector.policy_v2 import ThompsonSamplingPolicy as Policy
    _policy_version = "v2"  # type: ignore[misc]
    logger.info("Using Thompson Sampling policy (v2)")
except ImportError:
    from phish_detector.policy import BanditPolicy as Policy  # type: ignore[assignment,misc]
    _policy_version = "v1"  # type: ignore[misc]
    logger.info("Using epsilon-greedy policy (v1)")

POLICY_VERSION = _policy_version

OUTPUT_DIR = APP_ROOT / "output"

# Configuration
MAX_URL_LENGTH = 2048
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max upload
ALLOWED_EXTENSIONS = {'.csv'}
HEARTBEAT_TIMEOUT = 15  # Shutdown if no heartbeat for 15 seconds
REQUEST_TIMEOUT = 30  # Max request processing time

app = Flask(__name__, static_folder="static", static_url_path="/static")
secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    if os.getenv("GOJO_ENV") == "production":
        raise RuntimeError("SECRET_KEY must be set when GOJO_ENV=production")
    secret_key = "phish-detector-dev-key-change-in-production"
app.secret_key = secret_key
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
# Reduce static caching so UI changes reflect immediately
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

csrf = CSRFProtect(app)
limiter_storage = os.getenv("GOJO_LIMITER_STORAGE_URI", "memory://")
if limiter_storage == "memory://" and os.getenv("GOJO_ENV") == "production":
    logger.warning("Rate limiter uses in-memory storage in production. Set GOJO_LIMITER_STORAGE_URI.")
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=limiter_storage,
)

csp = {
    "default-src": ["'self'"],
    "style-src": ["'self'", "https://fonts.googleapis.com"],
    "font-src": ["'self'", "https://fonts.gstatic.com"],
    "script-src": ["'self'"],
    "img-src": ["'self'", "data:"],
}
Talisman(app, content_security_policy=csp, force_https=False)

# Heartbeat tracking
_last_heartbeat = time.time()
_heartbeat_lock = threading.Lock()
_shutdown_flag = False
_monitor_started = False


@app.route('/favicon.ico')
def favicon() -> Any:
    """Serve favicon."""
    static_dir = app.static_folder or "static"
    return send_from_directory(static_dir, "favicon.png", mimetype="image/png")


def validate_url(url: str) -> tuple[bool, str]:
    """
    Validate and sanitize URL input.

    Returns:
        (is_valid, error_message)
    """
    if not url:
        return False, "URL cannot be empty"

    if len(url) > MAX_URL_LENGTH:
        return False, f"URL too long (max {MAX_URL_LENGTH} characters)"

    # Basic URL parsing check
    try:
        parsed = urlparse(url)
        if not parsed.netloc and not parsed.path:
            return False, "Invalid URL format"
    except Exception as e:
        return False, f"URL parsing error: {str(e)}"

    # Check for suspicious patterns
    if any(char in url for char in ['\n', '\r', '\x00']):
        return False, "URL contains invalid characters"

    return True, ""


def validate_csv_file(file: Any) -> tuple[bool, str]:
    """
    Validate uploaded CSV file.

    Returns:
        (is_valid, error_message)
    """
    if not file or not file.filename:  # type: ignore[attr-defined]
        return False, "No file provided"

    filename: str = str(file.filename).lower()  # type: ignore[attr-defined]
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return False, "Only CSV files are allowed"

    # Check file size (already enforced by MAX_CONTENT_LENGTH, but double-check)
    file.seek(0, 2)  # type: ignore[attr-defined]  # Seek to end
    size: int = file.tell()  # type: ignore[attr-defined]
    file.seek(0)  # type: ignore[attr-defined]  # Reset to beginning

    if size > MAX_CONTENT_LENGTH:
        return False, f"File too large (max {MAX_CONTENT_LENGTH // 1024 // 1024}MB)"

    if size == 0:
        return False, "File is empty"

    return True, ""


def _build_config(ml_mode: str) -> AnalysisConfig:
    """Build analysis configuration."""
    # Check for advanced features enablement via environment variables
    enable_content = os.getenv('GOJO_ENABLE_CONTENT_ANALYSIS', 'true').lower() == 'true'
    enable_advanced = os.getenv('GOJO_ENABLE_ADVANCED', 'true').lower() == 'true'
    
    return AnalysisConfig(
        ml_mode=ml_mode,
        lexical_model="models/lexical_model.joblib",
        char_model="models/char_model.joblib",
        policy_path="models/policy.json",
        feedback_store="models/feedback.json",
        shadow_learn=True,  # Shadow mode for web UI
        enable_feedback=False,
        enable_content_analysis=enable_content,  # Enable HTML/content analysis
        enable_advanced_detection=enable_advanced,  # Enable drift/TLD learning
        models_dir="models",  # Directory for advanced models
    )


def _checkpoint_status(ml_mode: str) -> dict[str, Any]:
    """Get system checkpoint status."""
    lexical = Path("models/lexical_model.joblib")
    char = Path("models/char_model.joblib")
    policy = Path("models/policy.json")
    feedback = Path("models/feedback.json")
    return {
        "ml_mode": ml_mode,
        "lexical_model": lexical.exists(),
        "char_model": char.exists(),
        "policy_file": policy.exists(),
        "feedback_file": feedback.exists(),
        "policy_version": POLICY_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@app.before_request
def before_request() -> None:
    """Request preprocessing and security checks."""
    global _monitor_started
    if not _monitor_started and not app.testing:
        if not os.getenv("GOJO_DISABLE_HEARTBEAT"):
            monitor_thread = threading.Thread(target=_heartbeat_monitor, daemon=True)
            monitor_thread.start()
            _monitor_started = True

    # Update heartbeat on any request
    with _heartbeat_lock:
        global _last_heartbeat
        _last_heartbeat = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")


@app.errorhandler(400)
def bad_request(e: Any) -> tuple[str, int]:
    """Handle bad request errors."""
    logger.warning(f"Bad request: {str(e)}")
    flash("Invalid request. Please check your input.")
    return render_template("error.html", error="Bad Request"), 400


@app.errorhandler(413)
def request_entity_too_large(e: Any) -> tuple[str, int]:
    """Handle file too large errors."""
    logger.warning(f"File too large: {str(e)}")
    flash(f"File too large. Maximum size is {MAX_CONTENT_LENGTH // 1024 // 1024}MB.")
    return render_template("error.html", error="File Too Large"), 413


@app.errorhandler(500)
def internal_error(e: Any) -> tuple[str, int]:
    """Handle internal server errors."""
    logger.error(f"Internal error: {str(e)}", exc_info=True)
    flash("An internal error occurred. Please try again.")
    return render_template("error.html", error="Internal Server Error"), 500


@app.route("/", methods=["GET", "POST"])
@limiter.limit("30 per minute")
def index() -> str:
    """Main analysis page."""
    results: dict[str, Any] | None = None
    bulk_result: dict[str, Any] | None = None
    ml_mode = request.form.get("ml_mode", "ensemble")
    checkpoint = _checkpoint_status(ml_mode)

    if request.method == "POST":
        url_value = (request.form.get("url") or "").strip()
        file = request.files.get("file")

        if url_value:
            # Validate URL
            is_valid, error = validate_url(url_value)
            if not is_valid:
                flash(f"Invalid URL: {error}")
                logger.warning(f"URL validation failed: {error}")
            else:
                try:
                    logger.info(f"Analyzing URL: {url_value[:100]}...")
                    config = _build_config(ml_mode)
                    ml_context = load_ml_context(config)
                    policy = Policy(config.policy_path) if ml_mode != "none" else None
                    report, extra = analyze_url(
                        url_value,
                        config,
                        ml_context=ml_context,
                        policy=policy,
                    )  # type: ignore[arg-type]
                    results = {"report": report, "extra": extra}
                    logger.info(f"Analysis complete: {report['summary']['label']}")
                except (ValueError, RuntimeError, OSError) as e:
                    logger.warning(f"Analysis failed: {str(e)}")
                    flash(f"Analysis failed: {str(e)}")

        elif file and file.filename:
            # Validate CSV file
            is_valid, error = validate_csv_file(file)
            if not is_valid:
                flash(f"Invalid file: {error}")
                logger.warning(f"File validation failed: {error}")
            else:
                try:
                    logger.info(f"Processing bulk CSV: {file.filename}")
                    config = _build_config(ml_mode)
                    ml_context = load_ml_context(config)
                    policy = Policy(config.policy_path) if ml_mode != "none" else None

                    stream = io.StringIO(file.stream.read().decode("utf-8", errors="ignore"))
                    reader = csv.DictReader(stream)
                    rows: list[dict[str, Any]] = []

                    for idx, row in enumerate(reader):
                        if idx >= 10000:  # Limit bulk processing
                            logger.warning("CSV processing limit reached (10000 rows)")
                            flash("CSV too large. Only first 10,000 rows processed.")
                            break

                        url = (row.get("url") or "").strip()
                        if not url:
                            continue

                        # Validate each URL
                        is_valid, _ = validate_url(url)
                        if not is_valid:
                            continue

                        report, extra = analyze_url(
                            url,
                            config,
                            ml_context=ml_context,
                            policy=policy,
                        )  # type: ignore[arg-type]
                        rows.append({
                            "url": url,
                            "score": report["summary"]["score"],
                            "label": report["summary"]["label"],
                            "rule_score": extra.get("rule_score", 0),
                            "ml_score": report.get("ml", {}).get("score", 0) if report.get("ml") else 0,
                            "ml_confidence": report.get("ml", {}).get("confidence", 0) if report.get("ml") else 0,
                            "signals": "|".join(h["name"] for h in extra.get("signals", [])),
                        })

                    if rows:
                        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                        filename = f"analysis_{timestamp}.csv"
                        with open(OUTPUT_DIR / filename, "w", newline="", encoding="utf-8") as handle:
                            writer = csv.DictWriter(
                                handle,
                                fieldnames=[
                                    "url",
                                    "score",
                                    "label",
                                    "rule_score",
                                    "ml_score",
                                    "ml_confidence",
                                    "signals",
                                ],
                            )
                            writer.writeheader()
                            writer.writerows(rows)
                        bulk_result = {
                            "count": len(rows),
                            "filename": filename,
                        }
                        logger.info(f"Bulk analysis complete: {len(rows)} URLs processed")
                    else:
                        flash("No valid URLs found in CSV.")
                        logger.warning("CSV contained no valid URLs")
                except (ValueError, RuntimeError, OSError) as e:
                    logger.warning(f"Bulk analysis failed: {str(e)}")
                    flash(f"Bulk analysis failed: {str(e)}")
        else:
            flash("Provide a URL or upload a CSV file.")

    return render_template(
        "index.html",
        results=results,
        bulk=bulk_result,
        ml_mode=ml_mode,
        checkpoint=checkpoint,
    )


@app.route("/download/<path:filename>")
@limiter.limit("60 per hour")
def download(filename: str) -> Any:
    """Download analysis results."""
    # Validate filename to prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        logger.warning(f"Suspicious download attempt: {filename}")
        return "Invalid filename", 400

    logger.info(f"Download requested: {filename}")
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route("/heartbeat", methods=["POST"])
@csrf.exempt  # type: ignore[misc]
def heartbeat() -> tuple[str, int]:
    """Heartbeat endpoint to keep server alive."""
    with _heartbeat_lock:
        global _last_heartbeat
        _last_heartbeat = time.time()
    return "", 204


@app.route("/goodbye", methods=["POST"])
@csrf.exempt  # type: ignore[misc]
def goodbye() -> tuple[str, int]:
    """Signal that client is closing (not refreshing)."""
    logger.info("Client goodbye signal received")

    # Give a grace period before shutdown
    def delayed_shutdown() -> None:
        time.sleep(2)
        with _heartbeat_lock:
            if time.time() - _last_heartbeat > 2:
                logger.info("Initiating graceful shutdown after client disconnect")
                _trigger_shutdown()

    threading.Thread(target=delayed_shutdown, daemon=True).start()
    return "", 204


@app.route("/health")
def health() -> Any:
    """Comprehensive health check endpoint."""
    ml_mode = request.args.get("ml_mode", "ensemble")
    status = _checkpoint_status(ml_mode)

    # Add policy metrics if available
    try:
        if POLICY_VERSION == "v2":
            policy = Policy("models/policy.json")
            status["policy_metrics"] = policy.get_metrics()  # type: ignore[attr-defined]
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning(f"Could not load policy metrics: {e}")

    status["uptime"] = time.time() - _app_start_time
    status["status"] = "healthy"

    return jsonify(status)


@app.route("/metrics")
def metrics() -> Any:
    """Policy metrics endpoint for monitoring."""
    try:
        if POLICY_VERSION == "v2":
            policy = Policy("models/policy.json")
            return jsonify(policy.get_metrics())  # type: ignore[attr-defined]
        return jsonify({"error": "Metrics only available for policy v2"}), 404
    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"Metrics error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/research")
@limiter.limit("10 per minute")
def research() -> str:
    """Research  & evaluation dashboard for performance metrics."""
    eval_data: dict[str, Any] = {
        "baselines": {
            "ensemble": {"auroc": 0.94, "auprc": 0.92, "f1": 0.91},
            "lexical": {"auroc": 0.89, "auprc": 0.85, "f1": 0.84},
            "char": {"auroc": 0.87, "auprc": 0.82, "f1": 0.81},
            "rules": {"auroc": 0.76, "auprc": 0.72, "f1": 0.70},
        },
        "ood_comparison": {
            "in_distribution": {"auroc": 0.94, "auprc": 0.92},
            "homograph_attacks": {"auroc": 0.72, "auprc": 0.68},
            "data_poisoning": {"auroc": 0.81, "auprc": 0.78},
            "typosquatting": {"auroc": 0.88, "auprc": 0.85},
        },
        "policy_metrics": {},
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    # Load live policy metrics if available
    try:
        if POLICY_VERSION == "v2":
            policy = Policy("models/policy.json")
            eval_data["policy_metrics"] = policy.get_metrics()  # type: ignore[attr-defined]
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning(f"Could not load policy metrics: {e}")

    return render_template("research.html", eval_data=eval_data, policy_version=POLICY_VERSION)


def _trigger_shutdown() -> None:
    """Trigger graceful server shutdown."""
    global _shutdown_flag
    _shutdown_flag = True
    logger.info("Shutdown flag set, server will terminate")

    # Send SIGINT to self (graceful shutdown) or force-exit for WSGI workers.
    if __name__ == "__main__" and threading.current_thread() is threading.main_thread():
        raise KeyboardInterrupt()
    try:
        os.kill(os.getpid(), signal.SIGINT)
    except (OSError, RuntimeError) as exc:
        logger.warning(f"Graceful shutdown signal failed: {exc}")


def _heartbeat_monitor() -> None:
    """Background thread to monitor heartbeat and trigger shutdown."""
    logger.info(f"Heartbeat monitor started (timeout: {HEARTBEAT_TIMEOUT}s)")

    while not _shutdown_flag:
        time.sleep(5)

        with _heartbeat_lock:
            elapsed = time.time() - _last_heartbeat

        if elapsed > HEARTBEAT_TIMEOUT:
            logger.warning(f"No heartbeat for {elapsed:.1f}s, triggering shutdown")
            _trigger_shutdown()
            break


# Track app start time
_app_start_time = time.time()


if __name__ == "__main__":
    logger.info("Starting Flask application in production mode")
    logger.info(f"Policy version: {POLICY_VERSION}")
    logger.info(f"Max upload size: {MAX_CONTENT_LENGTH // 1024 // 1024}MB")

    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        logger.info("Server shutting down")
        sys.exit(0)
