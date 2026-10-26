import json
import math
import os
import sys
import time
import tkinter as tk
import ctypes
import ctypes.wintypes
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "face_landmarker.task"
CALIBRATION_PATH = APP_DIR / "calibration.json"
KEYS_PATH = APP_DIR / "keys.json"
DEFAULT_KEYS = {"mouse": "M", "dwell": "D", "double_click": "B", "recalibrate": "R"}

def load_keys():
    keys = DEFAULT_KEYS.copy()
    try:
        data = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
        for action in keys:
            value = str(data.get(action, keys[action])).strip().upper()
            if len(value) == 1 and value.isalnum():
                keys[action] = value
    except Exception:
        pass
    # Invalid or duplicated configuration falls back to safe defaults.
    if len(set(keys.values())) != len(keys):
        return DEFAULT_KEYS.copy()
    return keys

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


def ensure_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        return
    print("Descargando modelo Face Landmarker...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if MODEL_PATH.stat().st_size < 100_000:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError("El modelo descargado parece estar incompleto.")


def lm_xy(landmarks, index, width, height):
    p = landmarks[index]
    return np.array([p.x * width, p.y * height], dtype=np.float32)


def mean_points(landmarks, indices, width, height):
    return np.mean(
        [lm_xy(landmarks, i, width, height) for i in indices],
        axis=0,
    ).astype(np.float32)


def clamp01(v):
    return max(0.0, min(1.0, float(v)))


def eye_gaze(landmarks, side, width, height):
    if side == "left":
        iris_ids = LEFT_IRIS
        x0_id, x1_id = LEFT_EYE_LEFT, LEFT_EYE_RIGHT
        top_id, bottom_id = LEFT_EYE_TOP, LEFT_EYE_BOTTOM
    else:
        iris_ids = RIGHT_IRIS
        x0_id, x1_id = RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT
        top_id, bottom_id = RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM

    iris = mean_points(landmarks, iris_ids, width, height)
    x0 = lm_xy(landmarks, x0_id, width, height)
    x1 = lm_xy(landmarks, x1_id, width, height)
    top = lm_xy(landmarks, top_id, width, height)
    bottom = lm_xy(landmarks, bottom_id, width, height)

    eye_w = float(np.linalg.norm(x1 - x0))
    eye_h = float(np.linalg.norm(bottom - top))
    if eye_w < 2 or eye_h < 1:
        return None

    # Normalized iris position inside the eye.
    gx = (iris[0] - x0[0]) / (x1[0] - x0[0] + 1e-6)
    gy = (iris[1] - top[1]) / (bottom[1] - top[1] + 1e-6)

    return clamp01(gx), clamp01(gy), eye_w, eye_h


def gaze_feature(landmarks, width, height):
    left = eye_gaze(landmarks, "left", width, height)
    right = eye_gaze(landmarks, "right", width, height)
    samples = [s for s in (left, right) if s is not None]

    if not samples:
        return None, 0.0

    # Use the average of both eyes for the first calibrated model.
    gx = float(np.mean([s[0] for s in samples]))
    gy = float(np.mean([s[1] for s in samples]))

    if len(samples) == 2:
        disagreement = abs(left[0] - right[0]) + abs(left[1] - right[1])
        confidence = clamp01(1.0 - disagreement * 3.0)
    else:
        confidence = 0.55

    # Center features around 0.5 for numerical stability.
    x = gx - 0.5
    y = gy - 0.5

    # Quadratic feature vector. Nine calibration points are enough to fit
    # a useful first-order/quadratic mapping without overfitting too badly.
    feature = np.array(
        [1.0, x, y, x*x, x*y, y*y],
        dtype=np.float64,
    )
    return feature, confidence


class EMA:
    def __init__(self, alpha=0.18):
        self.alpha = float(alpha)
        self.value = None

    def reset(self):
        self.value = None

    def update(self, value):
        value = np.asarray(value, dtype=np.float32)
        if self.value is None:
            self.value = value.copy()
        else:
            self.value += self.alpha * (value - self.value)
        return self.value.copy()


class Calibrator:
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.points = [
            (0.08, 0.08), (0.50, 0.08), (0.92, 0.08),
            (0.08, 0.50), (0.50, 0.50), (0.92, 0.50),
            (0.08, 0.92), (0.50, 0.92), (0.92, 0.92),
        ]
        self.samples_x = []
        self.samples_y = []
        self.coeff_x = None
        self.coeff_y = None

    def add(self, feature, target_x, target_y):
        self.samples_x.append(feature)
        self.samples_y.append((target_x, target_y))

    def fit(self):
        if len(self.samples_x) < 6:
            return False

        A = np.asarray(self.samples_x, dtype=np.float64)
        Y = np.asarray(self.samples_y, dtype=np.float64)

        self.coeff_x = np.linalg.lstsq(A, Y[:, 0], rcond=None)[0]
        self.coeff_y = np.linalg.lstsq(A, Y[:, 1], rcond=None)[0]
        return True

    def predict(self, feature):
        if self.coeff_x is None or self.coeff_y is None:
            return None

        x = float(feature @ self.coeff_x)
        y = float(feature @ self.coeff_y)

        return np.array(
            [
                np.clip(x, 0, self.screen_w - 1),
                np.clip(y, 0, self.screen_h - 1),
            ],
            dtype=np.float32,
        )

    def save(self):
        data = {
            "screen_w": self.screen_w,
            "screen_h": self.screen_h,
            "coeff_x": self.coeff_x.tolist(),
            "coeff_y": self.coeff_y.tolist(),
        }
        CALIBRATION_PATH.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def load(self):
        if not CALIBRATION_PATH.exists():
            return False
        try:
            data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
            if (
                int(data["screen_w"]) != self.screen_w
                or int(data["screen_h"]) != self.screen_h
            ):
                return False
            self.coeff_x = np.asarray(data["coeff_x"], dtype=np.float64)
            self.coeff_y = np.asarray(data["coeff_y"], dtype=np.float64)
            return True
        except Exception:
            return False


def move_real_cursor(x, y):
    """Move the native Windows cursor without adding another dependency."""
    try:
        import ctypes
        ctypes.windll.user32.SetCursorPos(int(round(x)), int(round(y)))
        return True
    except Exception:
        return False


def click_left_mouse(clicks=1):
    """Send one or more Windows left clicks using SendInput."""
    if os.name != "nt":
        print("ERROR: el clic nativo solo está implementado para Windows.")
        return False

    ULONG_PTR = ctypes.c_size_t

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", ctypes.c_ulong), ("u", INPUT_UNION)]

    INPUT_MOUSE = 0
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int]
        user32.SendInput.restype = ctypes.c_uint
        total_clicks = max(1, min(int(clicks), 2))
        for index in range(total_clicks):
            events = (INPUT * 2)()
            events[0].type = INPUT_MOUSE
            events[0].mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, 0)
            events[1].type = INPUT_MOUSE
            events[1].mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, 0)
            sent = user32.SendInput(2, events, ctypes.sizeof(INPUT))
            if sent != 2:
                error = ctypes.get_last_error()
                print(f"ERROR SendInput: clic {index + 1}, se enviaron {sent}/2 eventos; GetLastError={error}")
                return False
            if index + 1 < total_clicks:
                time.sleep(0.08)
        return True
    except Exception as exc:
        print(f"ERROR al enviar click con SendInput: {exc}")
        return False


def screen_size():
    # Windows: obtain the real primary monitor resolution without requiring
    # another dependency. Fallback to 1280x720 if unavailable.
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1280, 720


def _lerp_angle_deg(a, b, amount):
    """Interpolate angles while taking the shortest path."""
    delta = (b - a + 180.0) % 360.0 - 180.0
    return a + delta * amount


def draw_bubble(canvas, center, velocity, confidence, scale=1.0, state=None):
    """Draw a gaze bubble inspired by the soft, deforming gaze indicators
    seen in eye-tracking/game overlays.

    It is deliberately an overlay only: it does not move or click the real
    Windows cursor yet. The shape changes with gaze velocity and confidence.
    """
    if state is None:
        state = {}

    cx, cy = float(center[0]), float(center[1])
    speed = float(np.linalg.norm(velocity))
    speed01 = min(speed / 0.030, 1.0)

    # Keep a memory of the direction so the bubble does not snap its rotation
    # every frame when the user briefly stops looking around.
    desired_angle = math.degrees(math.atan2(float(velocity[1]), float(velocity[0])))
    if speed < 0.0025:
        desired_angle = float(state.get("angle", 0.0))
    previous_angle = float(state.get("angle", desired_angle))
    angle = _lerp_angle_deg(previous_angle, desired_angle, 0.16)
    state["angle"] = angle

    # Base size + confidence-based uncertainty.
    base = 51.0 * scale
    uncertainty = 1.0 + (1.0 - float(confidence)) * 0.95

    # Movement stretches the bubble in the direction of travel. At rest it
    # settles into a slightly wider-than-tall oval.
    rx = base * (1.02 + 0.82 * speed01) * uncertainty
    ry = base * (0.74 + 0.10 * (1.0 - speed01)) * uncertainty

    # Slight breathing motion keeps it organic rather than perfectly static.
    t = time.perf_counter()
    breathe = 1.0 + 0.035 * math.sin(t * 5.0)
    rx *= breathe
    ry *= 1.0 + 0.025 * math.sin(t * 4.2 + 1.4)

    # Build a softly deformed ellipse as a polygon. The radial deformation is
    # small, so it still reads as a clear gaze zone instead of a random blob.
    points = []
    n = 96
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    wobble_phase = t * 2.4
    for i in range(n):
        theta = (2.0 * math.pi * i) / n
        wobble = (
            1.0
            + 0.045 * math.sin(theta * 3.0 + wobble_phase)
            + 0.022 * math.sin(theta * 5.0 - wobble_phase * 0.7)
        )
        ex = rx * math.cos(theta) * wobble
        ey = ry * math.sin(theta) * wobble
        x = cx + ex * cos_a - ey * sin_a
        y = cy + ex * sin_a + ey * cos_a
        points.append((int(round(x)), int(round(y))))

    polygon = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))

    # Soft translucent body.
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [polygon], (105, 65, 150))
    cv2.addWeighted(overlay, 0.10 + 0.06 * float(confidence), canvas,
                    1.0 - (0.10 + 0.06 * float(confidence)), 0, canvas)

    # Outer and inner outlines create the soft "zone of influence" look.
    overlay = canvas.copy()
    cv2.polylines(overlay, [polygon], True, (215, 120, 255),
                  max(2, int(round(3.0 * scale))), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)

    # A second, slightly smaller contour gives the bubble more depth.
    inner_points = []
    for i in range(n):
        theta = (2.0 * math.pi * i) / n
        wobble = 1.0 + 0.030 * math.sin(theta * 4.0 + wobble_phase * 0.8)
        ex = rx * 0.86 * math.cos(theta) * wobble
        ey = ry * 0.86 * math.sin(theta) * wobble
        x = cx + ex * cos_a - ey * sin_a
        y = cy + ex * sin_a + ey * cos_a
        inner_points.append((int(round(x)), int(round(y))))
    inner_polygon = np.asarray(inner_points, dtype=np.int32).reshape((-1, 1, 2))

    overlay = canvas.copy()
    cv2.polylines(overlay, [inner_polygon], True, (180, 95, 230),
                  1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.48, canvas, 0.52, 0, canvas)

    # Gaze point: deliberately small so the bubble remains the main visual.
    cv2.circle(canvas, (int(round(cx)), int(round(cy))),
               max(3, int(round(4.5 * scale))), (245, 245, 245),
               -1, cv2.LINE_AA)

    # Tiny directional tail while moving. It helps communicate where the
    # gaze bubble is heading without becoming a cursor/arrow yet.
    if speed > 0.004:
        tail_len = 95.0 + 65.0 * speed01
        tx = int(round(cx + velocity[0] * tail_len))
        ty = int(round(cy + velocity[1] * tail_len))
        overlay = canvas.copy()
        cv2.line(overlay, (int(round(cx)), int(round(cy))), (tx, ty),
                 (225, 165, 250), max(1, int(round(2 * scale))), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)


def create_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    return FaceLandmarker.create_from_options(options)


_last_mediapipe_timestamp_ms = 0


def read_frame(cap, landmarker, frame_index=None):
    global _last_mediapipe_timestamp_ms
    ok, frame = cap.read()
    if not ok:
        return None, None

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # MediaPipe VIDEO mode requires timestamps to increase across the entire
    # lifetime of a landmarker, including before and after recalibration.
    timestamp_ms = max(time.monotonic_ns() // 1_000_000, _last_mediapipe_timestamp_ms + 1)
    _last_mediapipe_timestamp_ms = timestamp_ms
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    return frame, result


def draw_calibration_screen(screen_w, screen_h, target, progress, message):
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    tx, ty = int(target[0]), int(target[1])

    # Target ring.
    cv2.circle(canvas, (tx, ty), 28, (215, 120, 255), 3, cv2.LINE_AA)
    cv2.circle(canvas, (tx, ty), 8, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.circle(canvas, (tx, ty), 3, (215, 120, 255), -1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        "CALIBRACION DE MIRADA",
        (40, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        message,
        (40, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (195, 195, 195),
        1,
        cv2.LINE_AA,
    )

    bar_w = min(520, screen_w - 80)
    bar_x = 40
    bar_y = screen_h - 55
    cv2.rectangle(
        canvas,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + 14),
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        canvas,
        (bar_x, bar_y),
        (bar_x + int(bar_w * progress), bar_y + 14),
        (215, 120, 255),
        -1,
    )

    cv2.putText(
        canvas,
        "Mira SOLO el punto. No muevas la cabeza.",
        (40, screen_h - 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return canvas


def run_calibration(cap, landmarker, screen_w, screen_h):
    calibrator = Calibrator(screen_w, screen_h)

    window = "Eye Gaze Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_index = 0
    points = calibrator.points

    # Each point: 0.7 s settle + 1.0 s collection.
    settle_frames = 21
    collect_frames = 30

    for point_index, (nx, ny) in enumerate(points):
        target = np.array(
            [nx * (screen_w - 1), ny * (screen_h - 1)],
            dtype=np.float32,
        )

        # Countdown/settling phase.
        for i in range(settle_frames):
            try:
                frame, result = read_frame(cap, landmarker, frame_index)
            except Exception as exc:
                print(f"Error durante el seguimiento: {exc}")
                running = False
                cap.release()
                overlay.destroy()
                return
            frame_index += 1
            if frame is None:
                return False

            progress = i / max(1, settle_frames)
            canvas = draw_calibration_screen(
                screen_w,
                screen_h,
                target,
                (point_index + progress * 0.5) / len(points),
                f"Punto {point_index + 1}/9 — posicionate y mantene la mirada...",
            )
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyWindow(window)
                return False

        collected = []
        confidences = []

        for i in range(collect_frames):
            try:
                frame, result = read_frame(cap, landmarker, frame_index)
            except Exception as exc:
                print(f"Error durante el seguimiento: {exc}")
                running = False
                cap.release()
                overlay.destroy()
                return
            frame_index += 1
            if frame is None:
                return False

            feature = None
            confidence = 0.0
            if result.face_landmarks:
                feature, confidence = gaze_feature(
                    result.face_landmarks[0],
                    frame.shape[1],
                    frame.shape[0],
                )

            if feature is not None and confidence >= 0.35:
                collected.append(feature)
                confidences.append(confidence)

            progress = (i + 1) / collect_frames
            canvas = draw_calibration_screen(
                screen_w,
                screen_h,
                target,
                (point_index + 0.5 + progress * 0.5) / len(points),
                f"Punto {point_index + 1}/9 — capturando mirada...",
            )

            if collected:
                cv2.putText(
                    canvas,
                    f"Muestras: {len(collected)}/{collect_frames}",
                    (40, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (190, 190, 190),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyWindow(window)
                return False

        if len(collected) < 8:
            cv2.destroyWindow(window)
            raise RuntimeError(
                f"No se pudieron obtener suficientes datos en el punto {point_index + 1}. "
                "Mejorá la iluminación y mantené la cara visible."
            )

        # Median is much more resistant to blinks and small tracking jitter.
        median_feature = np.median(np.asarray(collected), axis=0)
        calibrator.add(
            median_feature,
            float(target[0]),
            float(target[1]),
        )

    if not calibrator.fit():
        cv2.destroyWindow(window)
        raise RuntimeError("No se pudo ajustar la calibración.")

    calibrator.save()

    # Brief success screen.
    end = time.time() + 1.2
    while time.time() < end:
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "CALIBRACION COMPLETA",
            (50, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Iniciando seguimiento...",
            (50, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (195, 195, 195),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)
        cv2.waitKey(1)

    cv2.destroyWindow(window)
    return True



class BubbleOverlay:
    """Transparent, click-through desktop overlay that draws only the gaze bubble."""
    MAGENTA = "#ff00ff"

    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.geometry(f"{screen_w}x{screen_h}+0+0")
        self.root.configure(bg=self.MAGENTA)
        try:
            self.root.attributes("-transparentcolor", self.MAGENTA)
        except tk.TclError:
            pass
        try:
            self.root.attributes("-alpha", 0.78)
        except tk.TclError:
            pass
        self.root.attributes("-topmost", True)
        self.canvas = tk.Canvas(
            self.root,
            bg=self.MAGENTA,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)
        # Keep the overlay local to the bubble instead of covering the whole
        # desktop. Even if Windows ignores click-through styles, the rest of
        # the screen remains fully interactive.
        self.window_w = 400
        self.window_h = 360
        self._make_click_through()
        self.root.update_idletasks()
        self.root.update()

    def _make_click_through(self):
        """Make the actual top-level Windows window transparent to mouse input."""
        if os.name != "nt":
            return

        user32 = ctypes.windll.user32
        HWND = ctypes.c_void_p
        LONG_PTR = ctypes.c_ssize_t
        GWL_EXSTYLE = -20
        GA_ROOT = 2
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_NOACTIVATE = 0x08000000
        WS_EX_TOOLWINDOW = 0x00000080
        HWND_TOPMOST = -1
        SWP_NOSIZE = 0x0001
        SWP_NOMOVE = 0x0002
        SWP_NOACTIVATE = 0x0010
        SWP_SHOWWINDOW = 0x0040

        # winfo_id() can refer to Tk's child/client HWND. Apply the styles to
        # the real top-level window so the full-screen overlay cannot capture
        # clicks meant for the desktop or another application.
        user32.GetAncestor.argtypes = [HWND, ctypes.c_uint]
        user32.GetAncestor.restype = HWND
        user32.GetWindowLongPtrW.argtypes = [HWND, ctypes.c_int]
        user32.GetWindowLongPtrW.restype = LONG_PTR
        user32.SetWindowLongPtrW.argtypes = [HWND, ctypes.c_int, LONG_PTR]
        user32.SetWindowLongPtrW.restype = LONG_PTR
        user32.SetWindowPos.argtypes = [
            HWND, HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint,
        ]
        user32.SetWindowPos.restype = ctypes.c_int

        child_hwnd = HWND(self.root.winfo_id())
        hwnd = user32.GetAncestor(child_hwnd, GA_ROOT) or child_hwnd
        exstyle = int(user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE))
        exstyle |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
        user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, exstyle)
        user32.SetWindowPos(
            hwnd, HWND(HWND_TOPMOST), 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )

    def clear(self):
        self.canvas.delete("all")

    def draw(self, center, velocity, confidence, scale=1.0, state=None, dwell_progress=0.0, dwell_radius=42.0, dwell_active=False):
        if state is None:
            state = {}
        self.canvas.delete("all")
        global_cx, global_cy = float(center[0]), float(center[1])
        left = max(0, min(self.screen_w - self.window_w, int(global_cx - self.window_w / 2)))
        top = max(0, min(self.screen_h - self.window_h, int(global_cy - self.window_h / 2)))
        self.root.geometry(f"{self.window_w}x{self.window_h}+{left}+{top}")
        cx, cy = global_cx - left, global_cy - top
        speed = float(np.linalg.norm(velocity))
        speed01 = min(speed / 0.030, 1.0)

        desired_angle = math.degrees(math.atan2(float(velocity[1]), float(velocity[0])))
        if speed < 0.0025:
            desired_angle = float(state.get("angle", 0.0))
        previous_angle = float(state.get("angle", desired_angle))
        angle = _lerp_angle_deg(previous_angle, desired_angle, 0.16)
        state["angle"] = angle

        # Same larger-bubble proportions as the working v0.3 build.
        base = 51.0 * scale
        uncertainty = 1.0 + (1.0 - float(confidence)) * 0.95
        rx = base * (1.02 + 0.82 * speed01) * uncertainty
        ry = base * (0.74 + 0.10 * (1.0 - speed01)) * uncertainty

        t = time.perf_counter()
        breathe = 1.0 + 0.035 * math.sin(t * 5.0)
        rx *= breathe
        ry *= 1.0 + 0.025 * math.sin(t * 4.2 + 1.4)

        points = []
        n = 72
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        wobble_phase = t * 2.4
        for i in range(n):
            theta = (2.0 * math.pi * i) / n
            wobble = (
                1.0
                + 0.045 * math.sin(theta * 3.0 + wobble_phase)
                + 0.022 * math.sin(theta * 5.0 - wobble_phase * 0.7)
            )
            ex = rx * math.cos(theta) * wobble
            ey = ry * math.sin(theta) * wobble
            x = cx + ex * cos_a - ey * sin_a
            y = cy + ex * sin_a + ey * cos_a
            points.extend((x, y))

        self.canvas.create_polygon(
            *points,
            fill="#6f3f9a",
            outline="#d7b3ff",
            width=3,
            smooth=True,
        )

        # Small center marker, useful while testing but subtle.
        self.canvas.create_oval(
            cx - 4, cy - 4, cx + 4, cy + 4,
            fill="#f0d9ff", outline="",
        )

        if dwell_active:
            r = max(18.0, float(dwell_radius))
            # Full dim ring remains visible even at 0%; the bright arc shows progress.
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline="#795b91", width=3,
            )
            if dwell_progress > 0.0:
                self.canvas.create_arc(
                    cx - r, cy - r, cx + r, cy + r,
                    start=90,
                    extent=-360.0 * float(dwell_progress),
                    outline="#f0d9ff",
                    width=6,
                    style=tk.ARC,
                )

    def show(self):
        self.root.deiconify()
        self.root.attributes("-topmost", True)
        self.root.update_idletasks()

    def hide(self):
        self.root.withdraw()

    def destroy(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass



class ClickModePanel:
    """Small clickable control panel, separate from the click-through bubble overlay."""
    def __init__(self, root, initial_double=False, on_toggle=None):
        self.on_toggle = on_toggle
        self.double_click = bool(initial_double)
        self.window = tk.Toplevel(root)
        self.window.title("BasicEyeClick v0.9 - Clic")
        self.window.geometry("245x105+20+80")
        self.window.resizable(False, False)
        self.window.attributes("-topmost", True)
        self.window.configure(bg="#20202a")
        self.window.protocol("WM_DELETE_WINDOW", self.window.withdraw)
        tk.Label(
            self.window, text="Modo de clic por mirada", bg="#20202a", fg="white",
            font=("Segoe UI", 10, "bold")
        ).pack(pady=(9, 5))
        self.button = tk.Button(
            self.window, command=self.toggle, font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0, cursor="hand2", padx=12, pady=5,
        )
        self.button.pack(fill="x", padx=12, pady=(0, 9))
        self.update_label()

    def update_label(self):
        if self.double_click:
            self.button.configure(text="DOBLE CLIC  ✓", bg="#6b4aa5", fg="white", activebackground="#7959b5")
        else:
            self.button.configure(text="CLIC SIMPLE", bg="#dddddd", fg="#222222", activebackground="#eeeeee")

    def toggle(self):
        self.double_click = not self.double_click
        self.update_label()
        if self.on_toggle:
            self.on_toggle(self.double_click)

    def show(self):
        self.window.deiconify()
        self.window.attributes("-topmost", True)
        self.window.lift()

    def destroy(self):
        try:
            self.window.destroy()
        except tk.TclError:
            pass

def key_down(vk):
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
    except Exception:
        return False


def main():
    ensure_model()
    keys = load_keys()
    force_recalibration = "--recalibrate" in sys.argv

    screen_w, screen_h = screen_size()
    print(f"Pantalla detectada: {screen_w}x{screen_h}")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    gaze_filter = EMA(alpha=0.18)
    velocity_filter = EMA(alpha=0.20)
    bubble_state = {}

    calibrator = Calibrator(screen_w, screen_h)
    calibrated = calibrator.load() and not force_recalibration

    mouse_control = False
    dwell_enabled = False
    double_click_mode = False
    dwell_start = None
    dwell_anchor = None
    last_click_time = 0.0
    dwell_progress = 0.0
    DWELL_SECONDS = 1.00
    DWELL_RADIUS = 65.0
    CLICK_COOLDOWN = 1.00

    frame_index = 0
    last_gaze = None
    last_time = time.perf_counter()
    velocity = np.zeros(2, dtype=np.float32)
    confidence = 0.0
    gaze_xy = None
    key_codes = {action: ord(value) for action, value in keys.items()}
    previous_keys = {code: False for code in key_codes.values()}
    previous_keys.update({0x1B: False, 0x77: False})
    running = True
    print("Teclas: mouse={mouse}, fijacion={dwell}, modo clic={double_click}, recalibrar={recalibrate}, salir=ESC, prueba clic=F8".format(**keys))

    overlay = BubbleOverlay(screen_w, screen_h)
    overlay.hide()

    with create_landmarker() as landmarker:
        if not calibrated:
            print("No hay calibracion valida. Se iniciara calibracion de 9 puntos.")
            overlay.hide()
            ok = run_calibration(cap, landmarker, screen_w, screen_h)
            if not ok:
                cap.release()
                overlay.destroy()
                return
            calibrator = Calibrator(screen_w, screen_h)
            calibrated = calibrator.load()

        overlay.show()

        def handle_key(vk):
            nonlocal mouse_control, dwell_enabled, dwell_anchor, dwell_start, dwell_progress
            nonlocal last_gaze, gaze_xy, velocity, calibrator, calibrated
            nonlocal running, double_click_mode

            pressed = key_down(vk)
            was_pressed = previous_keys.get(vk, False)
            previous_keys[vk] = pressed
            if not pressed or was_pressed:
                return

            if vk == 0x1B:  # ESC
                running = False
                return

            if vk == 0x77:  # F8: isolated click test at current cursor position
                point = ctypes.wintypes.POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
                print(f"Prueba F8: intentando clic en cursor actual X={point.x}, Y={point.y}...")
                # Hide the overlay during the test so it cannot be the click target.
                overlay.hide()
                overlay.root.update_idletasks()
                if click_left_mouse(2 if double_click_mode else 1):
                    print("SendInput aceptó el doble clic." if double_click_mode else "SendInput aceptó el clic simple.")
                overlay.show()
                return

            if vk == key_codes["double_click"]:
                double_click_mode = not double_click_mode
                print(f"Modo de clic: {'DOBLE CLIC' if double_click_mode else 'CLIC SIMPLE'}")

            elif vk == key_codes["mouse"]:
                mouse_control = not mouse_control
                if not mouse_control:
                    dwell_enabled = False
                    dwell_anchor = None
                    dwell_start = None
                    dwell_progress = 0.0
                print(f"Control del mouse: {'ON' if mouse_control else 'OFF'}")

            elif vk == key_codes["dwell"]:
                if mouse_control:
                    dwell_enabled = not dwell_enabled
                    dwell_anchor = None
                    dwell_start = None
                    dwell_progress = 0.0
                    print(f"Dwell click: {'ON' if dwell_enabled else 'OFF'}")
                else:
                    print("Activa primero M (mouse).")

            elif vk == key_codes["recalibrate"]:
                print("Recalibrando...")
                overlay.hide()
                ok = run_calibration(cap, landmarker, screen_w, screen_h)
                if ok:
                    calibrator = Calibrator(screen_w, screen_h)
                    calibrated = calibrator.load()
                    gaze_filter.reset()
                    velocity_filter.reset()
                    bubble_state.clear()
                    last_gaze = None
                    gaze_xy = None
                    dwell_anchor = None
                    dwell_start = None
                    dwell_progress = 0.0
                    overlay.show()
                else:
                    running = False

        def tick():
            nonlocal running
            nonlocal frame_index, last_gaze, last_time, velocity, confidence, gaze_xy
            nonlocal dwell_anchor, dwell_start, dwell_progress, last_click_time

            if not running:
                try:
                    cap.release()
                finally:
                    overlay.destroy()
                return

            for vk in previous_keys:
                handle_key(vk)
            if not running:
                cap.release()
                overlay.destroy()
                return

            try:
                frame, result = read_frame(cap, landmarker, frame_index)
            except Exception as exc:
                print(f"Error durante el seguimiento: {exc}")
                running = False
                cap.release()
                overlay.destroy()
                return
            frame_index += 1
            if frame is None:
                running = False
                cap.release()
                overlay.destroy()
                return

            h, w = frame.shape[:2]
            feature = None
            confidence = 0.0
            gaze_xy = None
            if result.face_landmarks:
                feature, confidence = gaze_feature(result.face_landmarks[0], w, h)

            if feature is not None:
                predicted = calibrator.predict(feature)
                if predicted is not None:
                    gaze_xy = gaze_filter.update(predicted)
                    now = time.perf_counter()
                    dt = max(now - last_time, 1e-3)
                    if last_gaze is not None:
                        raw_velocity = (gaze_xy - last_gaze) / dt
                        raw_velocity = raw_velocity / np.array([screen_w, screen_h], dtype=np.float32)
                        velocity = velocity_filter.update(raw_velocity)
                    else:
                        velocity = np.zeros(2, dtype=np.float32)
                    last_gaze = gaze_xy.copy()
                    last_time = now

                    if mouse_control:
                        move_real_cursor(gaze_xy[0], gaze_xy[1])

                    now_dwell = time.perf_counter()
                    if dwell_enabled and mouse_control:
                        if dwell_anchor is None:
                            dwell_anchor = gaze_xy.copy()
                            dwell_start = now_dwell
                        else:
                            distance = float(np.linalg.norm(gaze_xy - dwell_anchor))
                            if distance > DWELL_RADIUS:
                                dwell_anchor = gaze_xy.copy()
                                dwell_start = now_dwell
                            else:
                                dwell_progress = min(1.0, (now_dwell - dwell_start) / DWELL_SECONDS)
                                if dwell_progress >= 1.0 and now_dwell - last_click_time >= CLICK_COOLDOWN:
                                    # Temporarily remove the overlay from hit-testing
                                    # so the click always reaches the app underneath.
                                    overlay.hide()
                                    overlay.root.update_idletasks()
                                    clicked = click_left_mouse(2 if double_click_mode else 1)
                                    overlay.show()
                                    if clicked:
                                        click_label = "DOBLE CLIC" if double_click_mode else "CLIC"
                                        print(f"{click_label} enviado en X={int(gaze_xy[0])}, Y={int(gaze_xy[1])}")
                                    last_click_time = now_dwell
                                    dwell_start = None
                                    dwell_anchor = None
                                    dwell_progress = 0.0
                    else:
                        dwell_anchor = None
                        dwell_start = None
                        dwell_progress = 0.0

                    overlay.draw(
                        gaze_xy,
                        velocity,
                        confidence,
                        scale=min(screen_w, screen_h) / 900.0,
                        state=bubble_state,
                        dwell_progress=dwell_progress,
                        dwell_radius=DWELL_RADIUS,
                        dwell_active=(dwell_enabled and mouse_control),
                    )
                else:
                    overlay.clear()
            else:
                overlay.clear()

            overlay.root.after(10, tick)

        try:
            overlay.root.after(10, tick)
            overlay.root.mainloop()
        finally:
            cap.release()
            overlay.destroy()


if __name__ == "__main__":
    main()
