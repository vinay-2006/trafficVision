$p = Start-Process -FilePath "python" -ArgumentList "-m streamlit run streamlit_app.py --server.port 8501 --server.headless true" -WorkingDirectory "d:\traffic-vehicle-analytics" -PassThru -NoNewWindow
Write-Host "Started Streamlit PID $($p.Id)"

try {
    Write-Host "Waiting for Streamlit server to be ready on port 8501..."
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        try {
            $resp = Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) {
                $ready = $true
                Write-Host "Streamlit server is ready!"
                break
            }
        } catch {
            # continue waiting
        }
    }

    if (-not $ready) {
        Write-Error "Streamlit failed to start within 30 seconds"
        exit 1
    }

    Write-Host "Executing Playwright browser test..."
    python scripts/browser_test.py
} finally {
    Write-Host "Stopping Streamlit PID $($p.Id)..."
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    # Also stop any stray streamlit processes
    Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*streamlit*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Write-Host "Cleanup complete."
}
