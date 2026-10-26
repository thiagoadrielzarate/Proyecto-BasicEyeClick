@echo off
setlocal
cd /d "%~dp0"
echo Preparando compilador de EXE...
py -3 -m pip install --upgrade pyinstaller
if errorlevel 1 (
  echo No se pudo instalar PyInstaller. Verifica que Python 3 este instalado.
  pause
  exit /b 1
)
echo.
echo Generando BasicEyeClick v0.9.exe...
py -3 -m PyInstaller --noconfirm --clean --onedir --windowed --name "BasicEyeClick v0.9" --distpath . --workpath build --specpath build launcher.py
if errorlevel 1 (
  echo.
  echo Hubo un error al generar el EXE.
  pause
  exit /b 1
)
echo.
echo Listo. EyeTracker.exe ya esta creado en la carpeta EyeTracker.
pause
