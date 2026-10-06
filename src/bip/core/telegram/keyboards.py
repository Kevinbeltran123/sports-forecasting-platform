"""Inline keyboard builders for Telegram Bot v2.

Pure layout — no side effects, no IO. Built from the callback codec in
``handlers.py`` so the produced markup is decodable by the handlers
without round-trip state.
"""

from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from bip.core.telegram.handlers import encode_callback


# Stake offerings (% of bankroll). Two-rung default keeps cognitive load
# minimal: one "normal" rung and one "half-confidence" rung. Adjust here
# if the operator wants to tune.
STAKE_RUNGS_PCT: tuple[float, ...] = (1.5, 0.75)

# REMIND default delay (minutes). Visible on the button label.
REMIND_DEFAULT_MINUTES: int = 5


def build_pick_keyboard(
    pick_id: int, *, tier: int,
) -> InlineKeyboardMarkup:
    """Return the appropriate inline keyboard for a pick's tier.

    Tier 1: [PLACE 1.5%] [PLACE 0.75%] / [SKIP] [REMIND 5m]
    Tier 2: [PLACE 1.5%] [PLACE 0.75%] / [SKIP]
    Tier 3: [PROMOTE] [DISMISS]
    """
    if tier == 3:
        return InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "PROMOTE",
                callback_data=encode_callback("promote", pick_id),
            ),
            InlineKeyboardButton(
                "DISMISS",
                callback_data=encode_callback("dismiss", pick_id),
            ),
        ]])

    place_row = [
        InlineKeyboardButton(
            f"PLACE {pct:.2f}%",
            callback_data=encode_callback("place", pick_id, f"{pct}"),
        )
        for pct in STAKE_RUNGS_PCT
    ]
    action_row = [
        InlineKeyboardButton(
            "SKIP", callback_data=encode_callback("skip", pick_id),
        ),
    ]
    if tier == 1:
        action_row.append(
            InlineKeyboardButton(
                f"REMIND {REMIND_DEFAULT_MINUTES}m",
                callback_data=encode_callback(
                    "remind", pick_id, str(REMIND_DEFAULT_MINUTES),
                ),
            ),
        )
    return InlineKeyboardMarkup([place_row, action_row])


def build_burst_digest_keyboard(
    pick_ids: list[int],
) -> InlineKeyboardMarkup:
    """One [PLACE 1.5%] [SKIP] row per pick in a §E burst digest."""
    rows = []
    for pid in pick_ids:
        rows.append([
            InlineKeyboardButton(
                f"#{pid} PLACE {STAKE_RUNGS_PCT[0]:.2f}%",
                callback_data=encode_callback(
                    "place", pid, str(STAKE_RUNGS_PCT[0]),
                ),
            ),
            InlineKeyboardButton(
                f"#{pid} SKIP",
                callback_data=encode_callback("skip", pid),
            ),
        ])
    return InlineKeyboardMarkup(rows)
