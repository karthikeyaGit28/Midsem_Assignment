$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Create .venv and install requirements-core.txt first. See README.md.'
}
& $taskPython -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
