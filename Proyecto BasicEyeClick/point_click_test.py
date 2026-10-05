import random
import tkinter as tk


class PointClickGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Gaze PointClick - Test Game")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#11131a")
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<r>", lambda e: self.restart())
        self.root.bind("<R>", lambda e: self.restart())

        self.canvas = tk.Canvas(root, bg="#11131a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)

        self.score = 0
        self.total = 10
        self.target = None
        self.target_bbox = None
        self.target_center = None
        self.target_radius = 0
        # Invisible assist area: a click is accepted when the cursor is
        # anywhere inside the target plus this radius. This simulates the
        # gaze-bubble tolerance without changing the eye tracker itself.
        self.assist_radius = 90
        self.last_mouse = (0, 0)
        self.restart()

    def restart(self):
        self.score = 0
        self.spawn_target()

    def spawn_target(self):
        self.canvas.delete("target")
        self.target = None
        self.root.update_idletasks()
        w = max(800, self.canvas.winfo_width())
        h = max(600, self.canvas.winfo_height())

        # Keep targets away from the extreme top-left corner so the small
        # eye-tracker control window can be placed there during testing.
        margin_x = max(80, int(w * 0.07))
        margin_y = max(100, int(h * 0.12))
        radius = random.randint(32, 48)
        x = random.randint(margin_x + radius, w - margin_x - radius)
        y = random.randint(margin_y + radius, h - margin_y - radius)

        self.target_bbox = (x - radius, y - radius, x + radius, y + radius)
        self.target_center = (x, y)
        self.target_radius = radius
        self.canvas.create_oval(
            *self.target_bbox,
            fill="#9b5de5",
            outline="#e8c9ff",
            width=4,
            tags="target",
        )
        self.canvas.create_oval(
            x - radius * 0.38,
            y - radius * 0.38,
            x + radius * 0.38,
            y + radius * 0.38,
            fill="#d8a7ff",
            outline="",
            tags="target",
        )
        self.update_hud()

    def update_hud(self):
        self.canvas.delete("hud")
        w = self.canvas.winfo_width()
        self.canvas.create_text(
            28, 24,
            anchor="nw",
            text=f"POINT & CLICK TEST    {self.score}/{self.total}",
            fill="#f2eef8",
            font=("Segoe UI", 18, "bold"),
            tags="hud",
        )
        self.canvas.create_text(
            28, 54,
            anchor="nw",
            text="Mirá el objetivo y mantené la mirada. R = reiniciar · ESC = salir",
            fill="#aaa4b4",
            font=("Segoe UI", 11),
            tags="hud",
        )

    def on_motion(self, event):
        self.last_mouse = (event.x, event.y)

    def on_click(self, event):
        if self.target_center is None:
            return

        # Assistive hit detection: the visible target is only the center of
        # the clickable region. The accepted region is expanded by
        # assist_radius, representing the tolerance of the gaze bubble.
        tx, ty = self.target_center
        dx = event.x - tx
        dy = event.y - ty
        distance = (dx * dx + dy * dy) ** 0.5
        accepted_radius = self.target_radius + self.assist_radius

        if distance <= accepted_radius:
            assisted = distance > self.target_radius
            self.score += 1

            # Brief feedback when the click was accepted by the invisible
            # assist area rather than landing directly on the target.
            if assisted:
                self.canvas.delete("assist_feedback")
                self.canvas.create_text(
                    tx, ty - self.target_radius - 18,
                    text="✓ ZONA DE BURBUJA",
                    fill="#d8a7ff",
                    font=("Segoe UI", 11, "bold"),
                    tags="assist_feedback",
                )
                self.root.after(350, lambda: self.canvas.delete("assist_feedback"))

            if self.score >= self.total:
                self.canvas.delete("all")
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2,
                    self.canvas.winfo_height() // 2 - 30,
                    text="¡COMPLETADO!",
                    fill="#f2eef8",
                    font=("Segoe UI", 42, "bold"),
                )
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2,
                    self.canvas.winfo_height() // 2 + 35,
                    text="Presioná R para jugar de nuevo · ESC para salir",
                    fill="#aaa4b4",
                    font=("Segoe UI", 16),
                )
                self.target_bbox = None
                self.target_center = None
                self.target_radius = 0
            else:
                self.spawn_target()


if __name__ == "__main__":
    root = tk.Tk()
    PointClickGame(root)
    root.mainloop()
