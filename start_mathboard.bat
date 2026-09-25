@echo off
rem Starts the Mathboard server and opens the board in a chromeless app window.
rem Extra arguments pass through, e.g.  start_mathboard.bat --model qwen3-vl:4b-instruct
rem
rem Recognizer: MATHBOARD_BACKEND is qwen (Ollama, the default), unimumer or ensemble. Set it here or in
rem mathboard.local.bat next to this file. unimumer and ensemble need llama.cpp's llama-server and the
rem Uni-MuMER files: this script starts llama-server in this console and stops it when the board stops.
rem Setup, paths and overrides: docs/BOARD_BACKENDS.md.
setlocal
cd /d "%~dp0"
rem Full path: with NoDefaultCurrentDirectoryInExePath set, cmd won't run a script from the current folder.
if exist "%~dp0mathboard.local.bat" call "%~dp0mathboard.local.bat"
if not defined MATHBOARD_BACKEND set MATHBOARD_BACKEND=qwen
set LLAMA_PID=
if /i "%MATHBOARD_BACKEND%"=="qwen" goto run

if not defined LLAMA_SERVER set "LLAMA_SERVER=%USERPROFILE%\tools\llama.cpp\llama-server.exe"
if not defined UNIMUMER_MODEL set "UNIMUMER_MODEL=%USERPROFILE%\models\uni-mumer\Uni-MuMER-Qwen3-VL-2B.Q8_0.gguf"
if not defined UNIMUMER_MMPROJ set "UNIMUMER_MMPROJ=%USERPROFILE%\models\uni-mumer\Uni-MuMER-Qwen3-VL-2B.mmproj-f16.gguf"
if not defined UNIMUMER_PORT set UNIMUMER_PORT=8792
set "MATHBOARD_UNIMUMER=http://127.0.0.1:%UNIMUMER_PORT%"

rem Already running (left over, or started by hand)? Use it as it is.
netstat -ano | findstr /r /c:"127\.0\.0\.1:%UNIMUMER_PORT% .*LISTENING" >nul && goto run

for %%f in ("%LLAMA_SERVER%" "%UNIMUMER_MODEL%" "%UNIMUMER_MMPROJ%") do if not exist %%f (
  echo [mathboard] %%~f not found, so Uni-MuMER can't start. Using qwen. See docs\BOARD_BACKENDS.md.
  set MATHBOARD_BACKEND=qwen
  set MATHBOARD_UNIMUMER=
  goto run
)

rem Uni-MuMER's training settings: images of 64 to 256 tokens (65,536 to 262,144 px), 2048-token context.
set "LLAMA_ARGS=-m "%UNIMUMER_MODEL%" --mmproj "%UNIMUMER_MMPROJ%" --host 127.0.0.1 --port %UNIMUMER_PORT% -ngl 999 -c 2048 -np 1 -fit off --cache-ram 0 --image-min-tokens 64 --image-max-tokens 256"
set "LLAMA_LOG=%TEMP%\mathboard-llama-server"
rem -NoNewWindow keeps llama-server on this console, so Ctrl+C or closing the window stops it too.
rem The PID goes through a file: llama-server would inherit a for /f pipe and hold it open.
del "%LLAMA_LOG%.pid" >nul 2>&1
powershell -NoProfile -Command "(Start-Process -FilePath $env:LLAMA_SERVER -ArgumentList $env:LLAMA_ARGS -NoNewWindow -PassThru -RedirectStandardOutput ($env:LLAMA_LOG + '.out.log') -RedirectStandardError ($env:LLAMA_LOG + '.log')).Id | Out-File -Encoding ascii ($env:LLAMA_LOG + '.pid')"
if exist "%LLAMA_LOG%.pid" set /p LLAMA_PID=<"%LLAMA_LOG%.pid"
if not defined LLAMA_PID (
  echo [mathboard] llama-server did not start. Using qwen.
  set MATHBOARD_BACKEND=qwen
  set MATHBOARD_UNIMUMER=
  goto run
)
echo [mathboard] llama-server started (PID %LLAMA_PID%, log %LLAMA_LOG%.log)

:run
python server.py --backend %MATHBOARD_BACKEND% %*
if defined LLAMA_PID taskkill /PID %LLAMA_PID% /F >nul 2>&1
endlocal
