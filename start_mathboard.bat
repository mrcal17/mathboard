@echo off
rem Starts the Mathboard server and opens the board in a chromeless app window.
rem Extra arguments pass through, e.g.  start_mathboard.bat --model qwen3-vl:4b-instruct
cd /d "%~dp0"
python server.py %*
