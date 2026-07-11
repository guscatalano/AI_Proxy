"""PyInstaller entry point for the frozen `ai-proxy` binary.

Uses an absolute import (not the package-relative __main__.py) so it works as a
standalone PyInstaller entry script.
"""
from ai_proxy import main

if __name__ == "__main__":
    main()
