param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArguments
)
$ErrorActionPreference = 'Stop'
$StudyRoot = Split-Path -Parent $PSScriptRoot
$ProjectBase = Split-Path -Parent (Split-Path -Parent $StudyRoot)
$SourceRoot = Join-Path $ProjectBase 'storage\sources\al-emadi\al-emadi-r001\original'
if (-not (Test-Path -LiteralPath (Join-Path $SourceRoot 'Binary_Drone_Audio'))) {
    throw 'Registered Al-Emadi source is missing.'
}
$StudyPython = Join-Path $ProjectBase 'local\environments\study-v3\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $StudyPython)) { throw 'Study V3 Python environment is missing.' }
$PreviousSourceRoot = $env:DRONE_AUDIO_DATASET_ROOT
try {
    $env:DRONE_AUDIO_DATASET_ROOT = $SourceRoot
    Push-Location -LiteralPath $StudyRoot
    try { & $StudyPython @PythonArguments; $StudyExitCode = $LASTEXITCODE }
    finally { Pop-Location }
}
finally { $env:DRONE_AUDIO_DATASET_ROOT = $PreviousSourceRoot }
exit $StudyExitCode
