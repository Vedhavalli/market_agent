# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


root = Path(SPECPATH)
datas = [(str(path), ".") for path in root.glob("*.py") if path.name != "desktop_app.py"]
datas.append((str(root / "config.toml"), "."))
binaries = []
hiddenimports = [
    "advice",
    "agent",
    "ai",
    "data",
    "explain",
    "fundamentals",
    "horizons",
    "indicators",
    "macro",
    "page_analysis",
    "page_screener",
    "portfolio",
    "report",
    "screener",
    "strategy",
    "swing",
    "util",
]

for package in ("streamlit", "streamlit_searchbox", "webview"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

analysis = Analysis(
    [str(root / "desktop_app.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
archive = PYZ(analysis.pure)
executable = EXE(
    archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="MarketAgent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    name="MarketAgent",
)