"""
Flask application factory for the Personal Knowledge Assistant.

This module creates the Flask app and registers all API blueprints.
Business logic lives in the blueprint modules under ``blueprints/``.

Blueprint organisation (Single Responsibility Principle):
    core          — health, info, frontend routes, document management
    chat          — RAG-powered Q&A, streaming, conversation memory
    search        — semantic search and document upload/ingestion
    study         — study paths, quiz generation, summarisation
    intelligence  — knowledge graph, conflicts, similarity, evaluation
    analytics     — query history, usage analytics, annotations

Usage:
    python -m src.web.app
"""

import time
from pathlib import Path

from flask import Flask, g, request

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Resolve template and static directories relative to this file
_WEB_DIR = Path(__file__).parent
_TEMPLATE_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"


def create_app() -> Flask:
    """Application factory for the Flask app.

    Creates the Flask instance, configures it, and registers all
    API blueprints. This pattern allows easy testing with different
    configurations.

    Returns:
        Configured Flask application.
    """
    application = Flask(
        __name__,
        template_folder=str(_TEMPLATE_DIR),
        static_folder=str(_STATIC_DIR),
    )

    # Configuration
    application.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

    # Register blueprints
    from .blueprints.core import core_bp
    from .blueprints.chat import chat_bp
    from .blueprints.search import search_bp
    from .blueprints.study import study_bp
    from .blueprints.intelligence import intelligence_bp
    from .blueprints.analytics import analytics_bp

    application.register_blueprint(core_bp)
    application.register_blueprint(chat_bp)
    application.register_blueprint(search_bp)
    application.register_blueprint(study_bp)
    application.register_blueprint(intelligence_bp)
    application.register_blueprint(analytics_bp)

    # ── Request timing middleware ───────────────────────────────────
    # Logs response latency for every API request and adds an
    # X-Response-Time header so the frontend can display it.

    @application.before_request
    def _start_timer():
        g.start_time = time.perf_counter()

    @application.after_request
    def _add_timing(response):
        if hasattr(g, "start_time"):
            elapsed_ms = (time.perf_counter() - g.start_time) * 1000
            response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"
            # Only log API calls, not static files
            if request.path.startswith("/api/"):
                logger.info(
                    "%s %s — %.1fms",
                    request.method, request.path, elapsed_ms,
                )
        return response

    logger.info(
        "Flask app created with %d blueprints.",
        len(application.blueprints),
    )

    return application


# Create the default app instance for direct running
app = create_app()


if __name__ == "__main__":
    import os
    # Debug mode is on by default for local development but can be
    # disabled for any deployment-like context by setting
    # FLASK_DEBUG=0 (or false/no/off) in the environment.
    debug = os.environ.get("FLASK_DEBUG", "1").lower() not in (
        "0", "false", "no", "off", "",
    )
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(debug=debug, port=port, threaded=True)