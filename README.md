# BasicEyeClick v0.9 — lanzador con menú

## Primera preparación
1. Extraé toda la carpeta del proyecto en una ubicación fija.
2. Si todavía no existe `BasicEyeClick v0.9.exe`, ejecutá `build_exe.bat` una sola vez en Windows. Necesita Python 3 y conexión a internet para instalar PyInstaller.
3. Después abrí `BasicEyeClick v0.9/BasicEyeClick v0.9.exe` (el modo de compilación ahora genera una carpeta con sus archivos de soporte). El archivo `.bat` original queda disponible como alternativa.

## Menú inicial
- **Iniciar programa:** inicia el seguimiento ocular normal.
- **Recalibrar:** fuerza la calibración de nueve puntos antes de empezar.
- **Configurar teclas:** permite cambiar las letras de las cuatro funciones principales. Las nuevas teclas se guardan en `keys.json` y se aplican al próximo inicio del seguimiento.

## Teclas iniciales
- `M`: activar/desactivar movimiento del cursor.
- `D`: activar/desactivar clic por fijación (primero activar M).
- `B`: alternar entre clic simple y doble clic.
- `R`: recalibrar mientras el seguimiento está activo.
- `F8`: probar el clic en la posición actual del cursor.
- `ESC`: salir del seguimiento.

En Configurar teclas, asigná una letra diferente a cada función. `ESC` y `F8` quedan fijas por ahora.

## Dependencias y webcam
Al iniciar o recalibrar, el lanzador crea `.venv` si hace falta e instala/verifica las dependencias de `requirements.txt`, igual que el archivo por lotes anterior. Se necesita internet la primera vez para instalar dependencias y descargar el modelo Face Landmarker.

`app.py`, `requirements.txt`, `keys.json` y la carpeta `models` deben permanecer junto a `BasicEyeClick v0.9.exe`. No muevas el EXE fuera de la carpeta del proyecto.


## Solución de rutas Tcl/Tk
El lanzador limpia las variables de entorno Tcl/Tk temporales de PyInstaller antes de iniciar Python externo, para evitar el error `Can't find a usable init.tcl`.


## Nombre de la versión
El nombre del proyecto es **BasicEyeClick v0.9**. El nombre anterior se conservó únicamente en algunos nombres internos de funciones o comentarios heredados del desarrollo.
