# One Antigravity quota window: run the Gemini evals until the quota runs out.
# Scheduled every 5 hours (task "mm-bench agy window"); skips if a run is live.
$repo = Split-Path -Parent $PSScriptRoot
$live = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*evalsuite.py run*' }
Set-Location $repo
if ($live) {
    # A run started without --quota-exit idles on the quota forever; replace it
    # when its newest log ends on a quota wait, otherwise leave it working.
    $last = Get-ChildItem verify\agy_*.log | Sort-Object LastWriteTime | Select-Object -Last 1
    $tail = if ($last) { Get-Content $last.FullName -Tail 1 } else { '' }
    if ($tail -notmatch 'quota reached') { exit 0 }
    $live | ForEach-Object { taskkill /PID $_.ProcessId /F | Out-Null }
    Start-Sleep -Seconds 5
}
$log = Join-Path $repo ("verify\agy_window_" + (Get-Date -Format 'yyyyMMdd_HHmm') + ".log")
$py = 'C:\Users\jluo6\Desktop\mm-long-storytelling-bench\.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
& $py -u scripts/evalsuite.py run --eval zeroshot,compact,halluc --backend antigravity `
    --agy C:/Users/jluo6/AppData/Local/agy/bin/agy.exe --model agy/gemini-3.8-flash-medium `
    --halluc-sample 300 --workers 3 --timeout 1800 --retry-errors --quota-exit *> $log
