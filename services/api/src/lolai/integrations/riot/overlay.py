"""Always-on-top overlay: champ-select advice (automatic) + augment advice (hotkey).

Run:  python -m lolai.integrations.riot.overlay
Press F8 (or the button) while augment cards are on screen.
Play League in Borderless or Windowed mode so the overlay can sit on top.
"""

from __future__ import annotations

import io
import queue
import threading
import time
import tkinter as tk

import mss
from PIL import Image
from pynput import keyboard

from lolai.agents import advisor
from lolai.integrations.riot.champ_select import read_champ_select
from lolai.integrations.riot.lcu import Lcu
from lolai.integrations.riot.live import read_live_game

HOTKEY = "<f8>"
POLL_SECONDS = 1.5
SETTLE_SECONDS = 2.0   # wait for champ select to stop changing before asking the model
WIDTH, HEIGHT, BAR_HEIGHT = 440, 520, 28


def capture_screen() -> bytes:
    """Primary monitor as a JPEG, scaled to 1600px wide (plenty for reading cards)."""
    with mss.mss() as sct:
        raw = sct.grab(sct.monitors[1])
    img = Image.frombytes("RGB", raw.size, raw.rgb)
    if img.width > 1600:
        img = img.resize((1600, round(img.height * 1600 / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


class Overlay:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("LoL AI")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.93)
        self.root.overrideredirect(True)
        self.root.geometry(f"{WIDTH}x{HEIGHT}+40+120")
        self.root.configure(bg="#14161f")

        bar = tk.Frame(self.root, bg="#232637")
        bar.pack(fill="x")
        self.status = tk.Label(bar, text="Waiting for League…", fg="#9aa3c7", bg="#232637",
                               font=("Segoe UI", 9), anchor="w")
        self.status.pack(side="left", padx=8, pady=4, fill="x", expand=True)
        tk.Button(bar, text="Augments (F8)", command=self.request_augments, relief="flat",
                  bg="#3b4bdb", fg="white", font=("Segoe UI", 9)).pack(side="right", padx=4, pady=3)
        tk.Button(bar, text="–", command=self.toggle, relief="flat", bg="#232637", fg="white",
                  width=2).pack(side="right")
        for w in (bar, self.status):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag)

        self.text = tk.Text(self.root, wrap="word", bg="#14161f", fg="#e6e8f2", bd=0,
                            font=("Segoe UI", 10), padx=10, pady=8, state="disabled")
        self.text.pack(fill="both", expand=True)
        self._show_text("Champion advice appears automatically in champ select.\n\n"
                        "When augment cards show up in game, press F8.")

        self.ui_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.busy_augments = threading.Event()
        self.hidden = False
        self._dx = self._dy = 0
        self.root.after(100, self._drain)

    # ---- UI (main thread only)
    def _drag_start(self, e) -> None:
        self._dx, self._dy = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y()

    def _drag(self, e) -> None:
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def toggle(self) -> None:
        self.hidden = not self.hidden
        self.root.geometry(f"{WIDTH}x{BAR_HEIGHT if self.hidden else HEIGHT}")

    def _show_text(self, s: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", s)
        self.text.configure(state="disabled")

    def _drain(self) -> None:
        try:
            while True:
                kind, value = self.ui_queue.get_nowait()
                if kind == "status":
                    self.status.configure(text=value)
                else:
                    self._show_text(value)
                    if self.hidden:
                        self.toggle()
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    # ---- thread-safe setters
    def set_status(self, s: str) -> None:
        self.ui_queue.put(("status", s))

    def set_text(self, s: str) -> None:
        self.ui_queue.put(("text", s))

    # ---- augments (F8)
    def request_augments(self) -> None:
        if self.busy_augments.is_set():
            return
        self.busy_augments.set()
        threading.Thread(target=self._augments_worker, daemon=True).start()

    def _augments_worker(self) -> None:
        try:
            self.set_status("Reading augment cards…")
            shot = capture_screen()  # grab first: the cards must still be on screen
            t0 = time.time()
            advice = advisor.advise_augments(shot, read_live_game())
            self.set_text(advisor.format_augments(advice))
            self.set_status(f"Augments analysed in {time.time() - t0:.0f}s")
        except Exception as e:  # show any failure in the overlay rather than dying silently
            self.set_status("Augment analysis failed")
            self.set_text(f"Error: {e}")
        finally:
            self.busy_augments.clear()


def champ_select_loop(ui: Overlay) -> None:
    lcu = Lcu()
    last_key = analysed_key = None
    changed_at = 0.0
    while True:
        try:
            phase = lcu.phase()
            if phase is None:
                ui.set_status("League client not found")
            elif phase != "ChampSelect":
                ui.set_status(f"Client: {phase}")
                last_key = analysed_key = None
            else:
                cs = read_champ_select(lcu)
                if cs:
                    key = cs.key()
                    if key != last_key:
                        last_key, changed_at = key, time.time()
                        ui.set_status("Champ select: waiting for picks to settle…")
                    elif key != analysed_key and time.time() - changed_at >= SETTLE_SECONDS:
                        analysed_key = key
                        ui.set_status("Champ select: analysing…")
                        t0 = time.time()
                        advice = advisor.advise_champ_select(cs)
                        ui.set_text(advisor.format_champ_select(advice))
                        ui.set_status(f"Champ select advice ({time.time() - t0:.0f}s)")
        except Exception as e:
            ui.set_status("Champ select analysis failed")
            ui.set_text(f"Error: {e}")
            time.sleep(5)
        time.sleep(POLL_SECONDS)


def main() -> None:
    ui = Overlay()
    threading.Thread(target=champ_select_loop, args=(ui,), daemon=True).start()
    # request_augments only spawns a thread, so it is safe to call from the hotkey thread.
    keyboard.GlobalHotKeys({HOTKEY: ui.request_augments}).start()
    ui.root.mainloop()


if __name__ == "__main__":
    main()
