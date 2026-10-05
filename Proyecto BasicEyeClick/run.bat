@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Creando entorno virtual...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo.
        echo No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
)

echo [2/2] Instalando/verificando dependencias...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo Iniciando BasicEyeClick en modo overlay...
".venv\Scripts\python.exe" app.py

pause
