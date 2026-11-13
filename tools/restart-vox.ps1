# Restart Vox through vox.ps1, then wait for its startup banner in the REAL
# %LOCALAPPDATA%\vox\vox.log and print it (only the banner: never dictated text).
#
# Run it through tools\restart-vox.cmd, which is meant to be started OUTSIDE
# a sandboxed shell. Inside the Claude desktop app's MSIX container
# %LOCALAPPDATA% points at a private copy, so a Vox started from there would
# keep its audio, records and log in the wrong folder, and the banner this
# script waits for would never appear where it looks.
#
#   restart-vox.ps1                      restart, wait up to 150 s for the banner
#   restart-vox.ps1 -CheckOnly           no restart: print the newest banner
#   restart-vox.ps1 -CheckOnly -LogPath \\localhost\C$\Users\m\AppData\Local\vox\vox.log
#                                        the same, read through the admin share
#                                        (works from inside the container)
#
# Exit code 0 = a "=== Vox ready ===" newer than the restart was found (or,
# with -CheckOnly, any banner was found).
param(
    [switch]$CheckOnly,
    [string]$LogPath = ''
)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$voxLog = if ($LogPath) { $LogPath } else { Join-Path $env:LOCALAPPDATA 'vox\vox.log' }

function Get-Banner([string]$path, [datetime]$notBefore) {
    # The newest ready line at or after $notBefore, plus the indented banner
    # lines right after it ("YYYY-mm-dd HH:MM:SS,mmm " is 24 characters).
    if (-not (Test-Path $path)) { return $null }
    # The whole file: the banner sits at the top of a log that grows all day
    # (vox.log rolls over at about 512 KB, at startup only).
    $lines = @(Get-Content $path -Encoding UTF8)
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        $ln = $lines[$i]
        if ($ln -notlike '*=== Vox ready ===*' -or $ln.Length -lt 19) { continue }
        $t = [datetime]::MinValue
        if (-not [datetime]::TryParseExact($ln.Substring(0, 19), 'yyyy-MM-dd HH:mm:ss',
                [Globalization.CultureInfo]::InvariantCulture,
                [Globalization.DateTimeStyles]::None, [ref]$t)) { continue }
        if ($t -lt $notBefore) { return $null }
        $out = @($ln)
        for ($j = $i + 1; $j -lt $lines.Count; $j++) {
            $msg = if ($lines[$j].Length -gt 24) { $lines[$j].Substring(24) } else { '' }
            if ($msg.StartsWith('  ') -or $msg.StartsWith('>> Logging self-check')) {
                $out += $lines[$j]
            } else { break }
        }
        return $out
    }
    return $null
}

if ($CheckOnly) {
    $banner = Get-Banner $voxLog ([datetime]::MinValue)
    if ($banner) { $banner; exit 0 }
    "restart-vox: no '=== Vox ready ===' in $voxLog"
    exit 1
}

$started = Get-Date
"restart-vox: $($started.ToString('yyyy-MM-dd HH:mm:ss')) root=$root"
"restart-vox: LOCALAPPDATA=$env:LOCALAPPDATA"
if ($env:LOCALAPPDATA -like '*\Packages\*') {
    "restart-vox: WARNING - LOCALAPPDATA is inside an app package: this shell is sandboxed and Vox would not use its real folder."
}
& (Join-Path $root 'vox.ps1') restart

$deadline = $started.AddSeconds(150)
$banner = $null
while ((Get-Date) -lt $deadline -and -not $banner) {
    Start-Sleep -Seconds 3
    $banner = Get-Banner $voxLog $started.AddSeconds(-2)
}
& (Join-Path $root 'vox.ps1') status
if ($banner) {
    "restart-vox: Vox came back:"
    $banner
    exit 0
}
"restart-vox: no new '=== Vox ready ===' in $voxLog within 150 s"
exit 1
