@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the virtual environment and install requirements.txt first.
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.address localhost --server.port 8501
