"""Watch the augment selection screen: detect select / reroll / hide / show from mouse clicks,
each one confirmed by comparing the card area before and after the click.

A click alone says only *where* the player clicked. What happened is decided by the screen:

    click on card i       -> all cards gone                  => SELECT card i
    click on a reroll     -> one card changed, others not    => REROLL that card (re-read it)
    click on hide/show    -> all cards gone                  => HIDDEN
    click on hide/show    -> (while hidden) cards are back   => SHOWN
    any other click, or nothing changed                      => ignored (ordinary game clicks)

Comparison is plain pixel difference of small colour thumbnails (no AI model, a few ms).

Usually 3 cards are offered, sometimes 4; each has its own layout, picked from how many cards F8
read. Screen areas, as fractions of the primary monitor (set in local.env; check them with
    python -m lolai.integrations.riot.overlay --calibrate      (3 cards on screen)
    python -m lolai.integrations.riot.overlay --calibrate 4    (4 cards on screen)):
    AUGMENT_CARD_CENTERS=0.308,0.500,0.692          horizontal centre of each card, 3 cards
    AUGMENT_CARD_CENTERS_4=0.212,0.404,0.596,0.788  the same with 4 cards
    AUGMENT_CARD_WIDTH=0.167   AUGMENT_CARD_WIDTH_4=0.167   card width (3 / 4 cards)
    AUGMENT_CARD_Y=0.179,0.667            card top, bottom
    AUGMENT_REROLL_Y=0.688,0.727          reroll button top, bottom (centred under its card)
    AUGMENT_REROLL_WIDTH=0.040            reroll button width
    AUGMENT_HIDE_BOX=0.446,0.762,0.555,0.824   hide/show button: left, top, right, bottom
Set AUGMENT_WATCH_DEBUG=1 to print every click and the measured differences.
"""

from __future__ import annotations

import io
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from pathlib import Path

import mss
from dotenv import load_dotenv
from PIL import Image, ImageChops, ImageDraw, ImageStat

# The overlay reads all its settings (zones, hotkeys, capture region) from local.env at import time.
load_dotenv(Path(__file__).resolve().parents[6] / "local.env")  # same file web_api.py loads


def floats_from_env(name: str, default: list[float]) -> list[float]:
    try:
        values = [float(x) for x in os.environ[name].split(",")]
        return values if values and all(0 <= v <= 1 for v in values) else default
    except (KeyError, ValueError):
        return default


# Defaults measured on a 1920x1080 Mayhem augment screen (3 cards). The 4-card layout is an
# estimate (same card size and spacing, one more card) until calibrated with --calibrate 4.
CARD_CENTERS = floats_from_env("AUGMENT_CARD_CENTERS", [0.308, 0.500, 0.692])
CARD_WIDTH = floats_from_env("AUGMENT_CARD_WIDTH", [0.167])[0]
CARD_CENTERS_4 = floats_from_env("AUGMENT_CARD_CENTERS_4", [0.212, 0.404, 0.596, 0.788])
CARD_WIDTH_4 = floats_from_env("AUGMENT_CARD_WIDTH_4", [CARD_WIDTH])[0]
MAX_CARDS = 4
CARD_Y = floats_from_env("AUGMENT_CARD_Y", [0.179, 0.667])[:2]
REROLL_Y = floats_from_env("AUGMENT_REROLL_Y", [0.688, 0.727])[:2]
REROLL_WIDTH = floats_from_env("AUGMENT_REROLL_WIDTH", [0.040])[0]
_hide = floats_from_env("AUGMENT_HIDE_BOX", [0.446, 0.762, 0.555, 0.824])
HIDE_BOX = tuple(_hide) if len(_hide) == 4 else (0.446, 0.762, 0.555, 0.824)
LABEL_GAP = 0.006  # labels sit this far above the top of the cards
DEBUG = os.environ.get("AUGMENT_WATCH_DEBUG") == "1"

# Mean per-pixel difference (0..1) of a card thumbnail, before vs after a click. Measured in game
# (data/augment_watch.log): an untouched card stays at 0.000-0.001; a rerolled card moves only
# 0.03-0.09, because all cards share the same frame and dark background (only icon and text change).
NOISE = 0.003        # untouched cards stay below about this
CHANGED = 0.03       # a card gone (select / hide) or back (show) changes at least this much
REROLL_MIN = 0.015   # a rerolled card changes at least this much…
DOMINANT = 4.0       # …and at least this many times more than any other card
# Seconds after a click to look: a reroll animation can take a while, so keep looking a few times.
CHECK_DELAYS = (0.3, 0.6, 1.0, 1.6, 2.4)
_THUMB = (24, 36)

Rect = tuple[float, float, float, float]  # left, top, right, bottom as screen fractions


def layout(count: int) -> tuple[list[float], float]:
    """(card centres, card width) for a screen showing `count` cards (3 or 4)."""
    return (CARD_CENTERS_4, CARD_WIDTH_4) if count >= 4 else (CARD_CENTERS, CARD_WIDTH)


def card_count(count: int) -> int:
    """How many cards the layout for `count` cards can actually place."""
    return min(count, len(layout(count)[0]))


def card_rect(i: int, count: int = 3) -> Rect:
    centres, width = layout(count)
    c = centres[i]
    return (c - width / 2, CARD_Y[0], c + width / 2, CARD_Y[1])


def reroll_rect(i: int, count: int = 3) -> Rect:
    c = layout(count)[0][i]
    return (c - REROLL_WIDTH / 2, REROLL_Y[0], c + REROLL_WIDTH / 2, REROLL_Y[1])


def label_rect(i: int, height: float, count: int = 3, width: float | None = None) -> Rect:
    """Where the stats label for card i goes: centred just above the card (card width by default)."""
    centres, card_width = layout(count)
    width = width or card_width
    c, bottom = centres[i], CARD_Y[0] - LABEL_GAP
    return (c - width / 2, max(bottom - height, 0.0), c + width / 2, bottom)


def _inside(fx: float, fy: float, r: Rect) -> bool:
    return r[0] <= fx <= r[2] and r[1] <= fy <= r[3]


def zone_at(fx: float, fy: float, count: int = 3) -> tuple[str, int] | None:
    """("card", i) / ("reroll", i) / ("hide", -1) for a click at screen fractions, else None."""
    if _inside(fx, fy, HIDE_BOX):
        return "hide", -1
    for i in range(card_count(count)):
        if _inside(fx, fy, card_rect(i, count)):
            return "card", i
        if _inside(fx, fy, reroll_rect(i, count)):
            return "reroll", i
    return None


# ---- screen ---------------------------------------------------------------------------------

def _monitor(sct) -> dict:
    return sct.monitors[1]


def grab_screen() -> Image.Image:
    """The whole primary monitor, in physical pixels (same coordinates as the mouse hook)."""
    with mss.mss() as sct:
        raw = sct.grab(_monitor(sct))
    return Image.frombytes("RGB", raw.size, raw.rgb)


def screen_size() -> tuple[int, int, int, int]:
    """(left, top, width, height) of the primary monitor in physical pixels."""
    with mss.mss() as sct:
        m = _monitor(sct)
    return m["left"], m["top"], m["width"], m["height"]


def crop(img: Image.Image, r: Rect) -> Image.Image:
    w, h = img.size
    return img.crop((int(r[0] * w), int(r[1] * h), int(r[2] * w), int(r[3] * h)))


def thumbs(img: Image.Image, count: int) -> list[Image.Image]:
    """One small thumbnail per card, in the layout for `count` cards."""
    # Colour, not grayscale: two different cards can have the same brightness.
    return [crop(img, card_rect(i, count)).convert("RGB").resize(_THUMB) for i in range(card_count(count))]


def difference(a: Image.Image, b: Image.Image) -> float:
    """Mean per-pixel, per-channel difference, 0 (identical) .. 1."""
    means = ImageStat.Stat(ImageChops.difference(a, b)).mean
    return sum(means) / len(means) / 255


def card_jpeg(img: Image.Image, i: int, count: int = 3) -> bytes:
    buf = io.BytesIO()
    crop(img, card_rect(i, count)).save(buf, "JPEG", quality=90)
    return buf.getvalue()


# The augment name sits in this band of the card (fractions of the card's height), below the icon
# and including a second line for long names. Measured on 1920x1080 screenshots.
TITLE_BAND = tuple(floats_from_env("AUGMENT_TITLE_BAND", [0.30, 0.62])[:2])


def title_jpeg(img: Image.Image, i: int, count: int = 3) -> bytes:
    """Just the name area of card i: a small image the vision model reads quickly."""
    left, top, right, bottom = card_rect(i, count)
    height = bottom - top
    band = (left, top + height * TITLE_BAND[0], right, top + height * TITLE_BAND[1])
    buf = io.BytesIO()
    crop(img, band).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def draw_zones(img: Image.Image, count: int = 3, label_height: float = 0.05) -> Image.Image:
    """The screenshot with card (green), reroll (orange), hide/show (purple) and label (blue) areas,
    in the layout for `count` cards."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size

    def box(r: Rect, colour: str, text: str) -> None:
        d.rectangle((r[0] * w, r[1] * h, r[2] * w, r[3] * h), outline=colour, width=3)
        d.text((r[0] * w + 6, r[1] * h + 4), text, fill=colour)

    for i in range(card_count(count)):
        box(card_rect(i, count), "#32ff6a", f"card {i + 1}")
        box(reroll_rect(i, count), "#ffa630", f"reroll {i + 1}")
        box(label_rect(i, label_height, count), "#4aa8ff", "label")
    box(HIDE_BOX, "#c77dff", "hide/show")
    return out


# ---- watcher ------------------------------------------------------------------------------------

LOG_PATH = Path(__file__).resolve().parents[6] / "data" / "augment_watch.log"
STABLE = 0.006        # a card is done animating when two grabs 0.25s apart differ less than this
SETTLE_SECONDS = 3.0  # give up waiting for a rerolled card to settle after this


def log(message: str) -> None:
    """Always written to data/augment_watch.log (and printed with AUGMENT_WATCH_DEBUG=1)."""
    line = f"{time.strftime('%H:%M:%S')} {message}"
    if DEBUG:
        print(f"[watch] {line}")
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


@dataclass
class Events:
    """Callbacks, called from a background thread."""
    select: Callable[[int], None]
    reroll: Callable[[int, bytes], None]   # card index, JPEG of the new card's name area
    hidden: Callable[[], None]
    shown: Callable[[], None]
    note: Callable[[str], None] = lambda _message: None  # short status for the player


class SelectionWatcher:
    """Every check compares the cards right AFTER a click with a picture taken AT the click, so
    whatever was already on screen (a hovered, raised card; animated art) is the same in both."""

    def __init__(self, events: Events) -> None:
        self.events = events
        self._lock = threading.Lock()
        self._cards = 0         # cards being watched; 0 = not watching
        self._hidden = False
        self._generation = 0    # bumps on arm/disarm, so stale checks are dropped

    @property
    def armed(self) -> bool:
        return self._cards > 0

    def arm(self, cards: int) -> None:
        """Start watching: the cards are on screen now."""
        with self._lock:
            self._cards = card_count(min(cards, MAX_CARDS))
            self._hidden = False
            self._generation += 1
        log(f"armed: {self._cards} cards")

    def disarm(self) -> None:
        with self._lock:
            was = self._cards
            self._cards = 0
            self._generation += 1
        if was:
            log("disarmed")

    def on_click(self, x: int, y: int) -> None:
        """Mouse hook thread: physical screen coordinates of a left click (button down).

        Windows delivers the click to the game only after this hook returns, so the "before"
        picture is taken right here: it is guaranteed to show the screen before the game reacts.
        A full-screen grab takes a few tens of ms and only happens for clicks on augment buttons."""
        with self._lock:
            generation, hidden, cards = self._generation, self._hidden, self._cards
        if not cards:
            return
        left, top, width, height = screen_size()
        fx, fy = (x - left) / width, (y - top) / height
        zone = zone_at(fx, fy, cards)
        # Only the augment screen's own buttons matter; everything else is an ordinary game click.
        # While hidden, only the hide/show button does anything.
        if zone is None or (hidden and zone[0] != "hide"):
            return
        before = thumbs(grab_screen(), cards)
        log(f"click {zone[0]} {zone[1] + 1 if zone[1] >= 0 else ''} at ({fx:.3f}, {fy:.3f}) "
            f"cards={cards} hidden={hidden}")
        threading.Thread(target=self._check, args=(zone, hidden, generation, cards, before),
                         daemon=True).start()

    @staticmethod
    def _rerolled(diffs: list[float], clicked: int) -> bool:
        """True when the card under the clicked reroll button changed and the others did not.
        Only that card can be rerolled by that button, so a change elsewhere (a hover glow on
        another card) never counts."""
        if not 0 <= clicked < len(diffs):
            return False
        mine = diffs[clicked]
        rest = max((d for i, d in enumerate(diffs) if i != clicked), default=0.0)
        return mine >= REROLL_MIN and mine >= DOMINANT * max(rest, NOISE)

    def _current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation and self._cards > 0

    def _settled(self, card: int, generation: int, cards: int) -> Image.Image | None:
        """Wait until the rerolled card stops animating; returns that screenshot (None if stale)."""
        img = grab_screen()
        previous = thumbs(img, cards)[card]
        deadline = time.monotonic() + SETTLE_SECONDS
        while time.monotonic() < deadline:
            time.sleep(0.25)
            if not self._current(generation):
                return None
            img = grab_screen()
            now = thumbs(img, cards)[card]
            moving = difference(previous, now)
            if moving < STABLE:
                return img
            previous = now
        log(f"card {card + 1} still changing after {SETTLE_SECONDS}s; reading it anyway")
        return img

    def _check(self, zone: tuple[str, int], hidden: bool, generation: int, cards: int,
               before: list[Image.Image]) -> None:
        kind, index = zone
        waited = 0.0
        for delay in CHECK_DELAYS:
            time.sleep(delay - waited)
            waited = delay
            if not self._current(generation):
                return  # re-armed or disarmed meanwhile
            now = thumbs(grab_screen(), cards)
            diffs = [difference(a, b) for a, b in zip(before, now)]
            log(f"  +{delay}s diffs={[round(d, 3) for d in diffs]}")
            all_changed = all(d > CHANGED for d in diffs)

            if kind == "hide" and all_changed:
                with self._lock:
                    self._hidden = not hidden
                log("  -> shown" if hidden else "  -> hidden")
                (self.events.shown if hidden else self.events.hidden)()
                return
            if kind == "card" and all_changed:
                log(f"  -> select card {index + 1}")
                self.disarm()
                self.events.select(index)
                return
            if kind == "reroll" and self._rerolled(diffs, index):
                self.events.note(f"Card {index + 1} rerolled — waiting for it to settle…")
                img = self._settled(index, generation, cards)
                if img is None:
                    return
                log(f"  -> reroll card {index + 1}")
                self.events.reroll(index, title_jpeg(img, index, cards))
                return
            # nothing (yet) – look again after the next delay
        log("  -> no change recognised")
        if kind == "reroll":
            self.events.note(f"Reroll {index + 1} clicked but no card change seen")
