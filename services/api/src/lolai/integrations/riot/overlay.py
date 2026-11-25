"""Always-on-top overlay: champ-select advice (automatic) + augment advice (hotkey).

Run:  python -m lolai.integrations.riot.overlay
Press F8 (or the button) while augment cards are on screen.

Only a region of the screen is sent to the vision model. Check that it covers your augment cards:
      python -m lolai.integrations.riot.overlay --save-crop     (open the augment screen, wait 5s)
      -> writes data/augment_crop.jpg. Adjust with e.g.  AUGMENT_REGION=0.2,0.2,0.8,0.8  (left,top,right,bottom
         as fractions of the screen).
Play League in Borderless or Windowed mode so the overlay can sit on top.
"""

from __future__ import annotations

import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
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
IMAGE_WIDTH = 900      # max width of what the model sees; measured: 1600px took 2.7s, 1024px 1.6s


def _region_from_env() -> tuple[float, float, float, float]:
    """(left, top, right, bottom) as fractions of the screen where augment cards appear."""
    default = (0.10, 0.15, 0.90, 0.85)  # deliberately wide: a guess until calibrated with --save-crop
    try:
        l, t, r, b = (float(x) for x in os.environ["AUGMENT_REGION"].split(","))
        return (l, t, r, b) if 0 <= l < r <= 1 and 0 <= t < b <= 1 else default
    except (KeyError, ValueError):
        return default


AUGMENT_REGION = _region_from_env()


def capture_screen(region: tuple[float, float, float, float] = AUGMENT_REGION) -> bytes:
    """The augment-card part of the primary monitor as a JPEG, scaled down to at most IMAGE_WIDTH.

    A smaller image means fewer tokens for the vision model, and cropping keeps more detail per pixel."""
    with mss.mss() as sct:
        mon = sct.monitors[1]
        l, t, r, b = region
        raw = sct.grab({"left": mon["left"] + int(mon["width"] * l), "top": mon["top"] + int(mon["height"] * t),
                        "width": int(mon["width"] * (r - l)), "height": int(mon["height"] * (b - t))})
    img = Image.frombytes("RGB", raw.size, raw.rgb)
    if img.width > IMAGE_WIDTH:
        img = img.resize((IMAGE_WIDTH, round(img.height * IMAGE_WIDTH / img.width)), Image.LANCZOS)
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

        # Bottom strip: augments the player has recorded this game.
        mine_bar = tk.Frame(self.root, bg="#232637")
        mine_bar.pack(side="bottom", fill="x")
        self.mine = tk.Label(mine_bar, text="", fg="#e6e8f2", bg="#232637", font=("Segoe UI", 9),
                             anchor="w", justify="left", wraplength=WIDTH - 130)
        self.mine.pack(side="left", padx=8, pady=4, fill="x", expand=True)
        tk.Button(mine_bar, text="Clear", command=self.clear_picks, relief="flat", bg="#232637",
                  fg="#9aa3c7", font=("Segoe UI", 8)).pack(side="right", padx=2)
        tk.Button(mine_bar, text="Undo", command=self.undo_pick, relief="flat", bg="#232637",
                  fg="#9aa3c7", font=("Segoe UI", 8)).pack(side="right", padx=2)

        self.text = tk.Text(self.root, wrap="word", bg="#14161f", fg="#e6e8f2", bd=0,
                            font=("Segoe UI", 10), padx=10, pady=8, state="disabled")
        self.text.pack(fill="both", expand=True)
        self._show_text("Champion advice appears automatically in champ select.\n\n"
                        "When augment cards show up in game, press F8, then click the one you took.")
        # "I picked this one" buttons; packed above the text only while cards are on offer.
        self.pick_frame = tk.Frame(self.root, bg="#14161f")

        self.picked: list[str] = []   # augments chosen this game; reset when the next champ select starts
        self._refresh_picked()

        self.ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
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

    # ---- recorded augments
    def _refresh_picked(self) -> None:
        shown = " › ".join(self.picked) if self.picked else "none yet"
        self.mine.configure(text=f"My augments: {shown}")

    def _show_offered(self, names: list[str]) -> None:
        for child in self.pick_frame.winfo_children():
            child.destroy()
        if not names:
            self.pick_frame.pack_forget()
            return
        for name in names:
            tk.Button(self.pick_frame, text=f"I picked: {name}", anchor="w", relief="flat",
                      bg="#2c3050", fg="white", font=("Segoe UI", 9),
                      command=lambda n=name: self.pick(n)).pack(fill="x", padx=8, pady=1)
        self.pick_frame.pack(fill="x", before=self.text)

    def pick(self, name: str) -> None:
        if name not in self.picked:
            self.picked.append(name)
        self._refresh_picked()
        self._show_offered([])
        self.status.configure(text=f"Recorded: {name}")

    def undo_pick(self) -> None:
        if self.picked:
            self.picked.pop()
            self._refresh_picked()

    def clear_picks(self) -> None:
        self.picked.clear()
        self._refresh_picked()
        self._show_offered([])

    def _drain(self) -> None:
        try:
            while True:
                kind, value = self.ui_queue.get_nowait()
                if kind == "status":
                    self.status.configure(text=value)
                elif kind == "hide":
                    self.root.attributes("-alpha", 0.0)
                    self.root.update_idletasks()
                    value.set()
                elif kind == "show":
                    self.root.attributes("-alpha", 0.93)
                elif kind == "append":
                    self.text.configure(state="normal")
                    self.text.insert("end", value)
                    self.text.configure(state="disabled")
                    self.text.see("end")
                elif kind == "offered":
                    self._show_offered(value)
                elif kind == "new_game":
                    self.clear_picks()
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

    def append_text(self, s: str) -> None:
        self.ui_queue.put(("append", s))

    def set_offered(self, names: list[str]) -> None:
        self.ui_queue.put(("offered", names))

    def new_game(self) -> None:
        self.ui_queue.put(("new_game", None))

    # ---- augments (F8)
    def request_augments(self) -> None:
        if self.busy_augments.is_set():
            return
        self.busy_augments.set()
        threading.Thread(target=self._augments_worker, daemon=True).start()

    def _capture_without_overlay(self) -> bytes:
        """Hide this window while grabbing the screen: it shows earlier augment names, which the
        vision model would otherwise read as cards."""
        hidden = threading.Event()
        self.ui_queue.put(("hide", hidden))
        hidden.wait(timeout=1.0)
        time.sleep(0.05)  # let the window manager repaint without us
        try:
            return capture_screen()
        finally:
            self.ui_queue.put(("show", None))

    def _augments_worker(self) -> None:
        try:
            self.set_status("Taking screenshot…")
            shot = self._capture_without_overlay()  # grab first: the cards must still be on screen

            self.set_status("Reading augment cards…")
            with ThreadPoolExecutor(max_workers=1) as pool:
                game_future = pool.submit(read_live_game)  # runs while the vision call is busy
                names = advisor.read_augment_names(shot)   # small vision call
                game = game_future.result()                # almost always already finished

            cards = advisor.rank_augments(names, game)  # database ranking, shown immediately
            self.set_text(advisor.format_ranking(cards, game))
            if not cards:
                self.set_status("No known augment cards found")
                return
            self.set_offered([c["name"] for c in cards])
            self.set_status("Writing AI recommendation…")

            # AI recommendation, appended as it is generated
            for chunk in advisor.stream_augment_advice(cards, game, list(self.picked)):
                self.append_text(chunk)
            self.set_status("Augment analysis done")
        except Exception as e:  # show any failure in the overlay rather than dying silently
            self.set_status("Augment analysis failed")
            self.set_text(f"Error: {e}")
        finally:
            self.busy_augments.clear()


def _warm_augment_model() -> None:
    try:
        advisor.warm_up()
    except Exception:
        pass  # warming is only an optimisation; the first F8 will just load it instead


def champ_select_loop(ui: Overlay) -> None:
    lcu = Lcu()
    last_key = analysed_key = None
    changed_at = 0.0
    prev_phase = None
    while True:
        try:
            phase = lcu.phase()
            if phase == "ChampSelect" and prev_phase != "ChampSelect":
                ui.new_game()  # a new game is starting: forget last game's augments
            if phase == "InProgress" and prev_phase != "InProgress":
                # load the augment model now (loading screen) so the first F8 doesn't pay for it
                threading.Thread(target=_warm_augment_model, daemon=True).start()
            if phase is not None:
                prev_phase = phase
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


def save_crop() -> None:
    """Calibration: capture the augment region after a short delay and save it for inspection."""
    from pathlib import Path
    out = Path(__file__).resolve().parents[6] / "data" / "augment_crop.jpg"
    print(f"Region (left, top, right, bottom): {AUGMENT_REGION}. Open the augment screen now…")
    for i in range(5, 0, -1):
        print(i, end=" ", flush=True)
        time.sleep(1)
    data = capture_screen()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    img = Image.open(io.BytesIO(data))
    print(f"\nSaved {img.width}x{img.height} -> {out}\nAll three cards fully inside? If not, widen AUGMENT_REGION.")


def main() -> None:
    if "--save-crop" in sys.argv:
        return save_crop()
    ui = Overlay()
    threading.Thread(target=champ_select_loop, args=(ui,), daemon=True).start()
    # request_augments only spawns a thread, so it is safe to call from the hotkey thread.
    keyboard.GlobalHotKeys({HOTKEY: ui.request_augments}).start()
    ui.root.mainloop()


if __name__ == "__main__":
    main()
