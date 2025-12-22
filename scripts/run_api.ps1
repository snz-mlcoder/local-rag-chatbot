
$repoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $repoRoot

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
