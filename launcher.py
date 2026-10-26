import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
APP_PATH = APP_DIR / "app.py"
REQ_PATH = APP_DIR / "requirements.txt"
KEYS_PATH = APP_DIR / "keys.json"
DEFAULT_KEYS = {"mouse": "M", "dwell": "D", "double_click": "B", "recalibrate": "R"}
OPTIONS = [chr(c) for c in range(ord("A"), ord("Z") + 1) if chr(c) not in ("Q",)]


def child_environment():
    """Avoid passing PyInstaller's temporary Tcl/Tk paths to external Python."""
    env = os.environ.copy()
    for name in ("TCL_LIBRARY", "TK_LIBRARY", "TCLLIBRARY", "TKLIBRARY", "PYTHONHOME", "PYTHONPATH"):
        env.pop(name, None)
    return env


def read_keys():
    result = DEFAULT_KEYS.copy()
    try:
        data = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
        for action in result:
            val = str(data.get(action, result[action])).strip().upper()
            if len(val) == 1 and val.isalpha():
                result[action] = val
    except Exception:
        pass
    if len(set(result.values())) != len(result):
        return DEFAULT_KEYS.copy()
    return result


def write_keys(values):
    if len(set(values.values())) != len(values):
        raise ValueError("Cada función debe tener una tecla diferente.")
    KEYS_PATH.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")


class Launcher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BasicEyeClick v0.9")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        frame = ttk.Frame(self.root, padding=20)
        frame.grid(sticky="nsew")
        ttk.Label(frame, text="BasicEyeClick v0.9", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(frame, text="Elegí qué querés hacer", font=("Segoe UI", 10)).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 16))
        ttk.Button(frame, text="Iniciar programa", command=lambda: self.launch(False)).grid(row=2, column=0, columnspan=2, sticky="ew", pady=4, ipady=5)
        ttk.Button(frame, text="Recalibrar", command=lambda: self.launch(True)).grid(row=3, column=0, columnspan=2, sticky="ew", pady=4, ipady=5)
        ttk.Button(frame, text="Configurar teclas", command=self.configure_keys).grid(row=4, column=0, columnspan=2, sticky="ew", pady=4, ipady=5)
        self.status = ttk.Label(frame, text="Las dependencias se preparan al iniciar el programa.", wraplength=310)
        self.status.grid(row=5, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.root.winfo_width()) // 2
        y = (self.root.winfo_screenheight() - self.root.winfo_height()) // 2
        self.root.geometry(f"+{x}+{y}")

    def configure_keys(self):
        current = read_keys()
        win = tk.Toplevel(self.root)
        win.title("Configurar teclas")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()
        frame = ttk.Frame(win, padding=16)
        frame.grid(sticky="nsew")
        ttk.Label(frame, text="Elegí una letra distinta para cada función.", wraplength=300).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        labels = [("mouse", "Activar/desactivar movimiento (M)"), ("dwell", "Activar/desactivar clic por fijación (D)"), ("double_click", "Alternar clic simple/doble (B)"), ("recalibrate", "Recalibrar durante el uso (R)")]
        vars_ = {}
        for i, (key, label) in enumerate(labels, start=1):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w", padx=(0, 12), pady=5)
            var = tk.StringVar(value=current[key])
            vars_[key] = var
            ttk.Combobox(frame, textvariable=var, values=OPTIONS, state="readonly", width=5).grid(row=i, column=1, sticky="e", pady=5)
        ttk.Label(frame, text="ESC sigue cerrando el programa y F8 sigue probando el clic.", wraplength=300).grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 8))
        def save():
            values = {key: var.get().upper() for key, var in vars_.items()}
            try:
                write_keys(values)
            except ValueError as exc:
                messagebox.showerror("Teclas repetidas", str(exc), parent=win)
                return
            messagebox.showinfo("Configuración guardada", "Las nuevas teclas se aplicarán la próxima vez que inicies el seguimiento ocular.", parent=win)
            win.destroy()
        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=2, sticky="e", pady=(4, 0))
        ttk.Button(buttons, text="Cancelar", command=win.destroy).grid(row=0, column=0, padx=4)
        ttk.Button(buttons, text="Guardar", command=save).grid(row=0, column=1, padx=4)
        win.update_idletasks()
        x = (self.root.winfo_screenwidth() - win.winfo_width()) // 2
        y = (self.root.winfo_screenheight() - win.winfo_height()) // 2
        win.geometry(f"+{x}+{y}")

    def run_command(self, command, description):
        self.status.config(text=description)
        self.root.update_idletasks()
        result = subprocess.run(command, cwd=APP_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env=child_environment())
        if result.returncode != 0:
            detail = result.stdout[-5000:] if result.stdout else "No se recibió información adicional."
            messagebox.showerror("No se pudo preparar el programa", f"{description}\n\n{detail}", parent=self.root)
            return False
        return True

    def prepare_environment(self):
        venv_python = APP_DIR / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            try:
                py_launcher = subprocess.run(["py", "-3", "-m", "venv", str(APP_DIR / ".venv")], cwd=APP_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env=child_environment())
                py_code, py_output = py_launcher.returncode, py_launcher.stdout or ""
            except FileNotFoundError:
                py_code, py_output = 1, "No se encontró el comando py."
            if py_code != 0:
                # Fall back to a regular Python command if the Windows py launcher is unavailable.
                try:
                    fallback = subprocess.run(["python", "-m", "venv", str(APP_DIR / ".venv")], cwd=APP_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env=child_environment())
                    fallback_code, fallback_output = fallback.returncode, fallback.stdout or ""
                except FileNotFoundError:
                    fallback_code, fallback_output = 1, "Tampoco se encontró el comando python."
                if fallback_code != 0:
                    detail = py_output + "\n" + fallback_output
                    messagebox.showerror("Python no encontrado", "Instalá Python 3.10–3.12 y habilitá el lanzador de Python.\n\n" + detail[-3000:], parent=self.root)
                    return None
        if not self.run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], "Actualizando el instalador de paquetes..."):
            return None
        if not self.run_command([str(venv_python), "-m", "pip", "install", "-r", str(REQ_PATH)], "Instalando/verificando dependencias..."):
            return None
        return venv_python

    def launch(self, recalibrate):
        if not APP_PATH.exists() or not REQ_PATH.exists():
            messagebox.showerror("Archivos faltantes", "Asegurate de que app.py y requirements.txt estén junto a BasicEyeClick v0.9.exe.", parent=self.root)
            return
        python = self.prepare_environment()
        if not python:
            return
        args = [str(python), str(APP_PATH)]
        if recalibrate:
            args.append("--recalibrate")
        self.root.withdraw()
        try:
            subprocess.Popen(args, cwd=APP_DIR, creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0), env=child_environment())
            self.root.after(900, self.root.destroy)
        except Exception as exc:
            self.root.deiconify()
            messagebox.showerror("No se pudo iniciar", str(exc), parent=self.root)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Launcher().run()
