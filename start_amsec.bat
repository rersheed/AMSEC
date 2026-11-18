@echo off
cd /d "%~dp0"
if not exist ".venv311\Scripts\python.exe" (
  echo Create the Python 3.11 environment first:
  echo   py -3.11 -m venv .venv311
  echo   .venv311\Scripts\python -m pip install -r requirements.txt
  exit /b 1
)
".venv311\Scripts\python.exe" -m streamlit run app.py --server.address localhost --server.port 8501
