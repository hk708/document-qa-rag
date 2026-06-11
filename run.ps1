# run.ps1 — Load environment variables from .env and start the server.
# Usage: .\run.ps1

# Read each line from .env, skip comments and blank lines,
# and set each KEY=VALUE pair as an environment variable.
Get-Content .env | Where-Object { $_ -match "^\s*[^#]\S+=.+" } | ForEach-Object {
    $parts = $_ -split "=", 2          # split on first = only
    $key   = $parts[0].Trim()
    $value = $parts[1].Trim()
    [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
    Write-Host "Set $key"
}

Write-Host ""
Write-Host "Starting server..."
uvicorn app.main:app --reload
