"""AI Proxy — a transparent inspector and rule engine for AI API traffic.

Exposes the FastAPI ``app`` and the ``main`` console entry point so the package can be
launched as ``ai-proxy`` (console script), ``python -m ai_proxy``, or imported as
``ai_proxy.proxy:app`` by an ASGI server.
"""

from .proxy import app, main

__all__ = ["app", "main"]
