"""Always-on-top overlay: champ-select advice (automatic) + augment advice (hotkey).

Run:  python -m lolai.integrations.riot.overlay
Press F8 (or the button) while augment cards are on screen.

Only a region of the screen is sent to the vision model. Check that it covers your augment cards:
      python -m lolai.integrations.riot.overlay --save-crop     (open the augment screen, wait 5s)
      -> writes data/augment_crop.jpg. Adjust with e.g.  AUGMENT_REGION=0.2,0.2,0.8,0.8  (left,top,right,bottom
         as fractions of the screen).
Play League in Borderless or Windowed mode so the overlay can sit on top (nothing can draw over
exclusive Fullscreen).

In game, without leaving League:
  - clicking the overlay does NOT take focus from League, so your keys keep working;
  - your pick is recorded from the game itself (see below); there are no pick hotkeys.
    If F8 does nothing, see what actually arrives:  python -m lolai.integrations.riot.overlay --test-keys
  - drag the top bar to place the overlay (e.g. next to the augment cards); the spot is remembered.
  - after F8, a small label appears ABOVE EACH augment card (tier, win, pick, slot, synergy; the best
    card is outlined in gold). Labels are click-through. Then play the augment screen normally
    (augment_watch.py):
      * click a card in League      -> recorded as your pick; the labels disappear
      * click a card's reroll       -> that card is re-read and its label/stats updated
      * click hide/show             -> the labels hide / show with the cards
    Usually 3 cards are offered, sometimes 4; the layout is picked from how many cards F8 read.
    Calibrate each once, while that many augment cards are on screen:
        python -m lolai.integrations.riot.overlay --calibrate     -> data/augment_zones.jpg
        python -m lolai.integrations.riot.overlay --calibrate 4   -> data/augment_zones_4.jpg
    and adjust in local.env (fractions of the screen) until the boxes match (defaults: 1920x1080):
        AUGMENT_CARD_CENTERS=0.308,0.500,0.692   AUGMENT_CARD_WIDTH=0.167   (3 cards)
        AUGMENT_CARD_CENTERS_4=0.212,0.404,0.596,0.788   AUGMENT_CARD_WIDTH_4=0.167   (4 cards)
        AUGMENT_CARD_Y=0.179,0.667   AUGMENT_REROLL_Y=0.688,0.727   AUGMENT_REROLL_WIDTH=0.040
        AUGMENT_HIDE_BOX=0.446,0.762,0.555,0.824 (left, top, right, bottom)
        AUGMENT_LABEL_WIDTH (defaults to the card width of the layout shown)
    AUGMENT_WATCH_DEBUG=1 prints each click and what the screen check measured.
"""

from __future__ import annotations

import ctypes
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
import tkinter as tk

import mss
from PIL import Image
from pynput import keyboard, mouse

from lolai.agents import advisor
from lolai.integrations.riot import augment_watch
from lolai.integrations.riot.champ_select import read_champ_select
from lolai.integrations.riot.db import DB_PATH
from lolai.integrations.riot.lcu import Lcu
from lolai.integrations.riot.live import read_live_game

HOTKEY = "<f8>"
POSITION_FILE = DB_PATH.parent / "overlay_position.json"
TOPMOST_SECONDS = 2.0  # re-assert always-on-top: a borderless game can push itself above us
CARD_LABEL_SECONDS = 120  # card labels disappear after this if no pick is recorded
_LABEL_BG, _BEST = "#14161f", "#e9c46a"


# Label width; unset = the width of the cards in the layout shown (3 or 4 cards).
LABEL_WIDTH = (augment_watch.floats_from_env("AUGMENT_LABEL_WIDTH", []) or [None])[0]
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


# ---- Windows: stay on top of the game without stealing its focus ----------------------------
_GWL_EXSTYLE = -20
_WS_EX_TOPMOST, _WS_EX_TOOLWINDOW, _WS_EX_NOACTIVATE = 0x00000008, 0x00000080, 0x08000000
_WS_EX_TRANSPARENT, _WS_EX_LAYERED = 0x00000020, 0x00080000  # together: mouse clicks pass through
_HWND_TOPMOST = -1
_SWP_NOSIZE, _SWP_NOMOVE, _SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010


def _make_non_activating(root: tk.Misc, click_through: bool = False) -> int | None:
    """Clicks on the window still work, but never take keyboard focus from the game; also hides it
    from Alt+Tab and the taskbar. click_through=True: clicks go to whatever is underneath instead.
    Returns the window handle, or None when not on Windows."""
    if sys.platform != "win32":
        return None
    root.update_idletasks()
    user32 = ctypes.windll.user32
    hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
    style = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE) | _WS_EX_NOACTIVATE | _WS_EX_TOOLWINDOW | _WS_EX_TOPMOST
    if click_through:
        style |= _WS_EX_LAYERED | _WS_EX_TRANSPARENT
    user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, style)
    return hwnd


def _raise_topmost(hwnd: int) -> None:
    ctypes.windll.user32.SetWindowPos(hwnd, _HWND_TOPMOST, 0, 0, 0, 0,
                                      _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE)


def _load_position() -> tuple[int, int]:
    try:
        pos = json.loads(POSITION_FILE.read_text(encoding="utf-8"))
        return int(pos["x"]), int(pos["y"])
    except (OSError, ValueError, KeyError, TypeError):
        return 40, 120


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
        x, y = _load_position()
        self.root.geometry(f"{WIDTH}x{HEIGHT}+{x}+{y}")
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
            w.bind("<ButtonRelease-1>", self._save_position)

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
        self.text.tag_configure("table", font=("Consolas", 9), wrap="none")
        self._show_text("Champion advice appears automatically in champ select.\n\n"
                        "When augment cards show up in game, press F8. Then pick, reroll or hide "
                        "cards in League as usual: the overlay follows what you click.")
        # "I picked this one" buttons; packed above the text only while cards are on offer.
        self.pick_frame = tk.Frame(self.root, bg="#14161f")
        self.offered: list[str] = []

        self.picked: list[str] = []   # augments chosen this game; reset when the next champ select starts
        self._refresh_picked()

        self.ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy_augments = threading.Event()
        self.current_names: list[str] = []     # cards on offer, left to right (worker threads)
        self._present_gen = 0                   # bumps when the shown cards change (see _present)
        self._present_lock = threading.Lock()
        self._reroll_lock = threading.Lock()    # rerolls are read one at a time
        self.hidden = False
        self._dx = self._dy = 0
        # labels under the cards, left to right: (window, handle, text label)
        self.card_windows: list[tuple[tk.Toplevel, int | None, tk.Label]] = []
        self._label_count = 0  # card layout (3 or 4) the labels are placed for
        self._card_timer: str | None = None
        self.labels_visible = True
        self.last_game = None  # LiveGame from the last F8, reused when a card is rerolled
        self.watcher = augment_watch.SelectionWatcher(augment_watch.Events(
            select=self.pick_card,
            reroll=self._on_reroll,
            hidden=lambda: self.ui_queue.put(("labels_visible", False)),
            shown=lambda: self.ui_queue.put(("labels_visible", True)),
            note=self.set_status,
        ))
        self.hwnd = _make_non_activating(self.root)
        self.root.after(100, self._drain)
        if self.hwnd:
            self.root.after(int(TOPMOST_SECONDS * 1000), self._keep_on_top)

    # ---- UI (main thread only)
    def _keep_on_top(self) -> None:
        _raise_topmost(self.hwnd)
        for _, hwnd, _ in self.card_windows:
            if hwnd:
                _raise_topmost(hwnd)
        self.root.after(int(TOPMOST_SECONDS * 1000), self._keep_on_top)

    # ---- labels above the augment cards
    def _show_card_labels(self, labels: list[tuple[str, bool]], count: int) -> None:
        """labels: (text, is_best) per card, left to right, for a screen showing `count` cards
        (3 or 4: each has its own layout). Each label sits just above its card and is
        click-through: clicks reach the game. With the same number of cards as already shown, the
        labels are updated in place (no flicker); a label hidden for a reroll comes back."""
        labels = labels[:augment_watch.card_count(count)] if labels else []
        if not labels:
            self._hide_card_labels()
            return
        same = len(labels) == len(self.card_windows) and count == self._label_count
        if not same:
            self._hide_card_labels()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        width = int(sw * (LABEL_WIDTH or augment_watch.layout(count)[1]))
        for i, (text, best) in enumerate(labels):
            if same:
                win, _, label = self.card_windows[i]
            else:
                win = tk.Toplevel(self.root)
                win.overrideredirect(True)
                win.attributes("-topmost", True)
                win.attributes("-alpha", 0.9)
                label = tk.Label(win, bg=_LABEL_BG, justify="center", wraplength=width - 12)
                label.pack(fill="both", expand=True, padx=2, pady=2, ipady=3)
            win.configure(bg=_BEST if best else _LABEL_BG)  # gold outline on the best card
            label.configure(text=text, fg="#ffffff" if best else "#e6e8f2",
                            font=("Segoe UI", 10, "bold" if best else "normal"))
            win.update_idletasks()
            height = win.winfo_reqheight()
            left, top, _, _ = augment_watch.label_rect(i, height / sh, count, width / sw)
            win.geometry(f"{width}x{height}+{int(sw * left)}+{int(sh * top)}")
            if not same:
                self.card_windows.append((win, _make_non_activating(win, click_through=True), label))
            elif self.labels_visible:
                win.deiconify()  # e.g. the label hidden while its card was re-read after a reroll
        self._label_count = count
        self.labels_visible = True
        self._schedule_label_hide(CARD_LABEL_SECONDS)

    def _set_card_label_visible(self, index: int, visible: bool) -> None:
        """Hide / show one card's label (while that card is re-read after a reroll)."""
        if 0 <= index < len(self.card_windows):
            win = self.card_windows[index][0]
            win.deiconify() if visible and self.labels_visible else win.withdraw()

    def on_global_click(self, x, y, button, pressed) -> None:
        """Global mouse hook thread: left clicks go to the selection watcher, which decides whether
        they were a card pick, a reroll or hide/show (labels themselves are not clickable)."""
        if pressed and button == mouse.Button.left:
            self.watcher.on_click(x, y)

    def _set_labels_visible(self, visible: bool) -> None:
        """The player hid / showed the augment selection: do the same with the labels."""
        self.labels_visible = visible
        for win, _, _ in self.card_windows:
            win.deiconify() if visible else win.withdraw()
        if self.card_windows:
            self.status.configure(text="Augment cards shown" if visible else "Augment cards hidden")

    def _schedule_label_hide(self, seconds: float) -> None:
        if self._card_timer:
            self.root.after_cancel(self._card_timer)
        self._card_timer = self.root.after(int(seconds * 1000), self._expire_labels)

    def _expire_labels(self) -> None:
        self._card_timer = None
        self.watcher.disarm()
        self._hide_card_labels()

    def _hide_card_labels(self) -> None:
        if self._card_timer:
            self.root.after_cancel(self._card_timer)
            self._card_timer = None
        for win, _, _ in self.card_windows:
            win.destroy()
        self.card_windows.clear()
        self._label_count = 0

    def _drag_start(self, e) -> None:
        self._dx, self._dy = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y()

    def _drag(self, e) -> None:
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _save_position(self, _e=None) -> None:
        try:
            POSITION_FILE.parent.mkdir(parents=True, exist_ok=True)
            POSITION_FILE.write_text(json.dumps({"x": self.root.winfo_x(), "y": self.root.winfo_y()}))
        except OSError:
            pass  # remembering the spot is a convenience only

    def toggle(self) -> None:
        self.hidden = not self.hidden
        self.root.geometry(f"{WIDTH}x{BAR_HEIGHT if self.hidden else HEIGHT}")

    def _show_text(self, s: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        # Table lines (advisor.format_ranking) need a monospace font for the columns to line up.
        for line in s.splitlines(keepends=True):
            is_table = advisor.TABLE_SEP in line or "┼" in line
            self.text.insert("end", line, ("table",) if is_table else ())
        self.text.configure(state="disabled")

    # ---- recorded augments
    def _refresh_picked(self) -> None:
        shown = " › ".join(self.picked) if self.picked else "none yet"
        self.mine.configure(text=f"My augments: {shown}")

    def _show_offered(self, names: list[str]) -> None:
        self.offered = list(names)
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
        self.watcher.disarm()  # recorded (card clicked in game, or a button): stop watching this round
        if name not in self.picked:
            self.picked.append(name)
        self._refresh_picked()
        self._show_offered([])
        self._hide_card_labels()  # the cards are gone, so are their labels
        self.status.configure(text=f"Recorded: {name}")

    def _pick_index(self, index: int) -> None:
        if 0 <= index < len(self.offered):
            self.pick(self.offered[index])
        else:
            self.status.configure(text=f"Card {index + 1} picked in game, but only "
                                       f"{len(self.offered)} cards were read — press F8")

    def undo_pick(self) -> None:
        if self.picked:
            self.picked.pop()
            self._refresh_picked()

    def clear_picks(self) -> None:
        self.watcher.disarm()
        self.picked.clear()
        self._refresh_picked()
        self._show_offered([])
        self._hide_card_labels()

    def _drain(self) -> None:
        try:
            while True:
                kind, value = self.ui_queue.get_nowait()
                if kind == "status":
                    self.status.configure(text=value)
                elif kind == "hide":
                    self._hide_card_labels()  # old labels would be read as card text
                    self.root.attributes("-alpha", 0.0)
                    self.root.update_idletasks()
                    value.set()
                elif kind == "show":
                    self.root.attributes("-alpha", 0.93)
                elif kind == "card_labels":
                    self._show_card_labels(*value)
                elif kind == "labels_visible":
                    self._set_labels_visible(value)
                elif kind == "card_label_visible":
                    self._set_card_label_visible(*value)
                elif kind == "append":
                    self.text.configure(state="normal")
                    self.text.insert("end", value)
                    self.text.configure(state="disabled")
                    self.text.see("end")
                elif kind == "offered":
                    self._show_offered(value)
                elif kind == "pick_index":
                    self._pick_index(value)
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

    def pick_card(self, index: int) -> None:
        """Selection watcher thread: card `index` (0 = leftmost) was taken in game."""
        self.ui_queue.put(("pick_index", index))

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
            self.watcher.disarm()
            self.set_status("Taking screenshot…")
            shot = self._capture_without_overlay()  # grab first: the cards must still be on screen

            self.set_status("Reading augment cards…")
            with ThreadPoolExecutor(max_workers=1) as pool:
                game_future = pool.submit(read_live_game)  # runs while the vision call is busy
                names = advisor.read_augment_names(shot)   # small vision call
                game = game_future.result()                # almost always already finished
            self.last_game = game
            self._present(names, game, arm=True)
        except Exception as e:  # show any failure in the overlay rather than dying silently
            self.set_status("Augment analysis failed")
            self.set_text(f"Error: {e}")
        finally:
            self.busy_augments.clear()

    def _new_presentation(self) -> int:
        """Start a new presentation; any AI text still streaming for an older one stops."""
        with self._present_lock:
            self._present_gen += 1
            return self._present_gen

    def _present(self, names: list[str], game, arm: bool) -> int:
        """Rank the cards, show table + labels, optionally start the selection watcher, then stream
        the AI recommendation. Returns how many cards were recognised."""
        gen = self._new_presentation()
        owned = list(self.picked)
        cards = advisor.rank_augments(names, game, owned)  # database ranking, shown immediately
        self.current_names = [c["name"] for c in cards]
        self.set_text(advisor.format_ranking(cards, game, owned))
        if not cards:
            self.set_status("No known augment cards found")
            return 0
        self.set_offered(list(self.current_names))
        # Labels and click detection need to know which card is where on screen. That only holds
        # when every card read was recognised (3 or 4 of them): a dropped card shifts the rest.
        on_screen = len(names)
        if len(cards) == on_screen and on_screen in (3, augment_watch.MAX_CARDS):
            best = advisor.best_card_index(cards, owned)
            self.ui_queue.put(("card_labels", ([(advisor.card_label(c, owned), i == best)
                                                for i, c in enumerate(cards)], on_screen)))
            if arm:
                self.watcher.arm(on_screen)
        else:
            self.watcher.disarm()
            self.ui_queue.put(("card_labels", ([], 0)))
            augment_watch.log(f"layout unknown: read {on_screen} cards, recognised {len(cards)}")
            self.set_status(f"Read {on_screen} cards, recognised {len(cards)} — no labels this round; "
                            "use the 'I picked' buttons")
            arm = False
        self.set_status("Writing AI recommendation…")
        stream = advisor.stream_augment_advice(cards, game, owned)
        try:
            for chunk in stream:
                if gen != self._present_gen:
                    return len(cards)  # a reroll replaced these cards: stop, don't append stale advice
                self.append_text(chunk)
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()  # stops the model generating for nothing
        self.set_status("Augment analysis done — pick a card in game")
        return len(cards)

    def _on_reroll(self, index: int, title_jpeg: bytes) -> None:
        """Watcher thread: card `index` was rerolled. Read only that card's name (a small crop of
        its title area: fast), swap it in, and refresh everything.
        Never dropped: it also interrupts advice still being written, and several quick rerolls
        are handled one after another. The crop read is kept in data/last_reroll_card.jpg."""
        self._new_presentation()  # stop the advice for the old cards now, freeing the model
        # Its stats belong to the old card: hide that one label until the new card is read.
        self.ui_queue.put(("card_label_visible", (index, False)))

        def restore_label() -> None:
            self.ui_queue.put(("card_label_visible", (index, True)))

        def read(jpeg: bytes) -> str | None:
            _save_debug_image("last_reroll_card.jpg", jpeg)
            raw = advisor.read_card_name(jpeg)
            known = advisor.verify_augment_names([raw]) if raw else []
            augment_watch.log(f"  reroll card {index + 1}: read {raw!r} -> {known!r}")
            return known[0] if known else None

        def work() -> None:
            with self._reroll_lock:
                names = list(self.current_names)
                try:
                    if index >= len(names):
                        restore_label()
                        self.set_status(f"Rerolled card {index + 1} is not one of the cards shown — press F8")
                        return
                    self.set_status(f"Card {index + 1} rerolled — reading it…")
                    name = read(title_jpeg)
                    if name is None:  # caught mid-animation, or a misread: one more look, fresh
                        time.sleep(0.8)
                        name = read(augment_watch.title_jpeg(augment_watch.grab_screen(), index, len(names)))
                    if name is None:
                        restore_label()
                        self.set_status(f"Couldn't read rerolled card {index + 1} — press F8")
                        return
                    if name == names[index]:
                        augment_watch.log(f"  reroll card {index + 1}: still reads as {name!r}")
                        self.set_status(f"Reroll {index + 1}: the card still reads as {name!r}")
                    else:
                        augment_watch.log(f"  reroll card {index + 1}: {names[index]!r} -> {name!r}")
                    names[index] = name
                    self._present(names, self.last_game, arm=False)  # redraws all labels, this one included
                except Exception as e:
                    restore_label()
                    augment_watch.log(f"  reroll update failed: {e!r}")
                    self.set_status("Reroll update failed")
                    self.set_text(f"Error: {e}")

        threading.Thread(target=work, daemon=True).start()


def _save_debug_image(name: str, jpeg: bytes) -> None:
    """Keep the last image the vision model was given, in data/, to check what it saw."""
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        (DB_PATH.parent / name).write_bytes(jpeg)
    except OSError:
        pass


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


def _layout_arg() -> int:
    """`--calibrate 4` / `--test-labels 4`: the 4-card layout; anything else: 3 cards."""
    return 4 if "4" in sys.argv[1:] else 3


def calibrate(count: int = 3) -> None:
    """Screenshot of the augment screen with the card / reroll / label areas drawn on it."""
    suffix = "" if count == 3 else f"_{count}"
    out = DB_PATH.parent / f"augment_zones{suffix}.jpg"
    print(f"Open the augment selection screen now ({count} cards visible)…")
    for i in range(5, 0, -1):
        print(i, end=" ", flush=True)
        time.sleep(1)
    img = augment_watch.draw_zones(augment_watch.grab_screen(), count)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=90)
    print(f"\nSaved {img.width}x{img.height} -> {out}\n"
          "Green = card (click = pick), orange = reroll button, purple = hide/show, blue = where labels go.\n"
          "Green must cover each card, orange each reroll button, purple the hide/show button.\n"
          f"Adjust AUGMENT_CARD_CENTERS{suffix} / AUGMENT_CARD_WIDTH{suffix} / AUGMENT_CARD_Y / "
          "AUGMENT_REROLL_Y / AUGMENT_REROLL_WIDTH / AUGMENT_HIDE_BOX in local.env and rerun.")


def test_labels(count: int = 3, seconds: int = 30) -> None:
    """Show sample labels where the real ones will go, to line them up with the augment cards."""
    ui = Overlay()
    sample = "S · 60.3% win · 6.9% pick\nslot +1% · syn +4%"
    centres = augment_watch.layout(count)[0]
    ui._show_card_labels([(f"card {i + 1}\n{sample}", i == 1) for i in range(len(centres))], count)
    ui.set_text(f"Sample labels for {seconds}s ({count} cards), just above each card.\nCentres: {centres}  "
                f"Card top: {augment_watch.CARD_Y[0]}\n\n"
                "Adjust AUGMENT_CARD_CENTERS(_4) / AUGMENT_CARD_Y / AUGMENT_LABEL_WIDTH in local.env and rerun.")
    ui.root.after(seconds * 1000, ui.root.destroy)
    ui.root.mainloop()


def test_keys() -> None:
    """Print every key the global hook receives, to find out why F8 does nothing."""
    print("Hotkey: F8 = analyse the augment cards.")
    print("Press keys (also with League focused). Ctrl+C to stop.\n")

    def on_press(key):
        vk = getattr(key, "vk", None) or getattr(getattr(key, "value", None), "vk", None)
        print(f"  pressed {key!s:<20} vk={vk}", flush=True)

    with keyboard.Listener(on_press=on_press) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass


def main() -> None:
    if "--save-crop" in sys.argv:
        return save_crop()
    if "--test-labels" in sys.argv:
        return test_labels(_layout_arg())
    if "--test-keys" in sys.argv:
        return test_keys()
    if "--calibrate" in sys.argv:
        return calibrate(_layout_arg())
    ui = Overlay()
    threading.Thread(target=champ_select_loop, args=(ui,), daemon=True).start()
    # request_augments only spawns a thread, so it is safe to call from the hotkey thread.
    keyboard.GlobalHotKeys({HOTKEY: ui.request_augments}).start()
    # Clicks in game (card pick / reroll / hide): a global hook, because League keeps the mouse.
    mouse.Listener(on_click=ui.on_global_click).start()
    ui.root.mainloop()


if __name__ == "__main__":
    main()
