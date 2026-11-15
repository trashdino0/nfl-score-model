@echo off
cd /d "%~dp0"
python scripts\grade_last_week.py
echo.
pause
