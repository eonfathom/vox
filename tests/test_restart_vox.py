"""Checks for tools/restart-vox.ps1's refusal guard (Windows only).

It never runs the real vox.ps1: the script is copied next to a fake vox.ps1
that only records its arguments, in a temp folder.

Run:  .venv/Scripts/python.exe tests/test_restart_vox.py   (or pytest tests/)
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(HERE, "tools", "restart-vox.ps1")
IS_WINDOWS = sys.platform == "win32"

FAKE_VOX = """param([string]$a)
Add-Content -LiteralPath (Join-Path $PSScriptRoot 'calls.txt') -Value $a
"""


def _ps(args, env=None, timeout=60):
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass"] + args,
        capture_output=True, text=True, env=env, timeout=timeout)


def test_overridden_localappdata_refuses_without_calling_vox():
    # A LOCALAPPDATA pointed at a scratch folder (as tests and agents do, often
    # under the unredirected Temp) passes the share probe, so the script must
    # refuse on the folder itself: exit 2, vox.ps1 never called, no probe left.
    if not IS_WINDOWS:
        return
    base = tempfile.mkdtemp(prefix="restart-vox-test-")
    try:
        root = os.path.join(base, "root")
        os.makedirs(os.path.join(root, "tools"))
        shutil.copy(SCRIPT, os.path.join(root, "tools", "restart-vox.ps1"))
        with open(os.path.join(root, "vox.ps1"), "w", encoding="utf-8") as f:
            f.write(FAKE_VOX)
        la = os.path.join(base, "la")
        os.makedirs(os.path.join(la, "vox"))
        env = dict(os.environ, LOCALAPPDATA=la)
        p = _ps(["-File", os.path.join(root, "tools", "restart-vox.ps1")], env=env)
        assert p.returncode == 2, (p.returncode, p.stdout, p.stderr)
        assert "STOPPED" in p.stdout, p.stdout
        assert not os.path.exists(os.path.join(root, "calls.txt")), \
            open(os.path.join(root, "calls.txt")).read()
        left = [n for _, _, fs in os.walk(la) for n in fs
                if n.startswith(".restart-vox-probe-")]
        assert not left, left
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_default_localappdata_comparison():
    # Test-DefaultLocalAppData lifted out of the script through the AST, so
    # the probe (which writes a file) never runs here.
    if not IS_WINDOWS:
        return
    cases = [
        ("C:\\Users\\u\\AppData\\Local", "C:\\Users\\u\\AppData\\Local", True),
        ("C:\\Users\\u\\AppData\\Local\\", "C:\\Users\\u\\AppData\\Local", True),
        ("c:\\users\\U\\appdata\\local", "C:\\Users\\u\\AppData\\Local", True),
        ("C:\\Users\\u\\AppData\\Local\\Temp\\x", "C:\\Users\\u\\AppData\\Local", False),
        ("C:\\Users\\u\\AppData\\Local\\Temp", "C:\\Users\\u\\AppData\\Local", False),
        ("D:\\scratch", "C:\\Users\\u\\AppData\\Local", False),
        ("", "C:\\Users\\u\\AppData\\Local", False),
        ("C:\\Users\\u\\AppData\\Local", "", False),
    ]
    lines = [
        "$ast = [System.Management.Automation.Language.Parser]::ParseFile("
        "'%s', [ref]$null, [ref]$null)" % SCRIPT.replace("'", "''"),
        "$fn = $ast.FindAll({ param($n) $n -is "
        "[System.Management.Automation.Language.FunctionDefinitionAst] -and "
        "$n.Name -eq 'Test-DefaultLocalAppData' }, $true) | Select-Object -First 1",
        "if (-not $fn) { 'MISSING'; exit 3 }",
        ". ([scriptblock]::Create($fn.Extent.Text))",
    ]
    for la, known, _ in cases:
        lines.append("[string](Test-DefaultLocalAppData '%s' '%s')"
                     % (la.replace("'", "''"), known.replace("'", "''")))
    p = _ps(["-Command", "\n".join(lines)])
    assert p.returncode == 0, (p.returncode, p.stdout, p.stderr)
    got = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    want = [str(w) for _, _, w in cases]
    assert got == want, list(zip(cases, got))


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", name)
    print(f"{n} checks passed")
