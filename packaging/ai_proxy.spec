# PyInstaller spec — builds a single-file `ai-proxy` executable with the web UI
# and all Python deps baked in, so the npm package can ship it with zero runtime
# requirements on the user's machine.
#
# Build:  pyinstaller packaging/ai_proxy.spec --distpath dist/bin --workpath build/pyi
# (run from the repo root, in an env where `ai_proxy` and its deps are installed)

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Bundle the packaged web UI at ai_proxy/static (matches STATIC_DIR's _MEIPASS lookup).
datas = collect_data_files("ai_proxy", includes=["static/*"])

# uvicorn loads its loop/protocol/lifespan backends dynamically, so PyInstaller can't
# see them by static analysis — collect them (and anyio's backends) explicitly.
hiddenimports = (
    collect_submodules("uvicorn")
    + collect_submodules("anyio")
    + ["httptools", "websockets", "watchfiles", "click", "h11"]
)

a = Analysis(
    ["entry.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ai-proxy",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
