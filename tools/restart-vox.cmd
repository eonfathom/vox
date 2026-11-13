@echo off
rem Restart Vox from OUTSIDE a sandboxed shell, and log whether it came back.
rem
rem Why: a shell inside the Claude desktop app runs in its MSIX container,
rem where %LOCALAPPDATA% is redirected to a private copy. A Vox started from
rem there would keep its audio, dictation records and vox.log in that copy,
rem not in the real C:\Users\<you>\AppData\Local\vox. Explorer is outside the
rem container, so start this file through it:
rem
rem   powershell -NoProfile -Command "Start-Process explorer.exe 'C:\Users\m\dev\vox\tools\restart-vox.cmd'"
rem
rem From a normal terminal, running it directly is the same thing.
rem
rem It runs tools\restart-vox.ps1: vox.ps1 restart (guardian-aware), then waits
rem up to 150 s for a new "=== Vox ready ===" in the real vox.log and copies
rem that startup banner (only the banner) into tools\restart-vox.log, which is
rem git-ignored. The last line of that log is "exit 0" when Vox came back.
setlocal
set "LOG=%~dp0restart-vox.log"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0restart-vox.ps1" > "%LOG%" 2>&1
>> "%LOG%" echo exit %ERRORLEVEL%
endlocal
