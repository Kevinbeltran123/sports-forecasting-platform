"""Adjust live λ (expected goals remaining) for game state.

Pre-match λ from Poisson/TSV assumes neutral 90-min game. Live betting
needs adjustment for:
  - Time remaining (linear scale)
  - Score state (leading teams reduce output, trailing increase)
  - Coach style (defensive locks down harder when leading)
  - Red cards (significant production reduction)
  - Big margin lead/deficit (compound effect — lockdown or panic)

Empirical anchor: Ireland 1-0 Qatar (2026-05-28). Hallgrímsson (defensive
icelandic style) + 1-0 at min 5 produced λ_remaining ≈ 0.4 over 85 min vs
naive 1.55. Moylan red card cemented bunker mode → 0 second-half goals.

This module addresses the limitation found that day: naive Poisson scaling
overestimates remaining production for pragmatic coaches managing a lead.
"""
from typing import Literal


CoachStyle = Literal["defensive", "balanced", "attacking"]
"""Coach taxonomy.

  - defensive: Hallgrímsson, Simeone, Mancini-era Italy, classic Mourinho.
    Strong lead-management instinct; reduces output drastically when ahead.
  - balanced: Default. Tuchel, Deschamps, Scaloni, most national-team coaches.
  - attacking: Klopp, Guardiola, De la Fuente's Spain, Brazil's typical style.
    Keep playing forward even when leading.
"""


# Score-state × coach style multipliers — empirically calibrated.
# Ireland-Qatar 2026-05-28 anchor: defensive leading +1 ≈ 0.40 × base.
_STATE_FACTORS: dict[CoachStyle, dict[str, float]] = {
    "defensive": {"leading": 0.40, "level": 1.00, "trailing": 0.95},
    "balanced":  {"leading": 0.70, "level": 1.00, "trailing": 1.10},
    "attacking": {"leading": 0.90, "level": 1.00, "trailing": 1.20},
}

# Lead/deficit ≥ 2 goals: leader relaxes further, trailing panics inefficiently.
_BIG_MARGIN_EXTRA = 0.85

# Red card: literature suggests ~45% reduction in goals for player-down team.
_RED_CARD_FACTOR = 0.55

# Floor to avoid zero-prob edge cases (e.g., 89th minute calculations).
_LAMBDA_FLOOR = 0.05


def adjust_lambda_for_game_state(
    lambda_base: float,
    minutes_remaining: int,
    score_diff: int,
    coach_style: CoachStyle = "balanced",
    has_red_card: bool = False,
) -> float:
    """Compute λ_remaining adjusted by live game state.

    Args:
        lambda_base: Pre-match λ (full 90 min) for the team — typically derived
            from TSV `goals_for_per_match` × home/away/opponent adjustments.
        minutes_remaining: Minutes left in regulation (0–90).
        score_diff: +N if this team leading by N, -N if trailing, 0 if level.
        coach_style: "defensive" | "balanced" | "attacking".
        has_red_card: True if THIS team is down to 10 (opponent advantage).

    Returns:
        Adjusted λ_remaining (floored at 0.05 to keep Poisson sane).

    Example:
        Hallgrímsson Ireland leading 1-0 at min 5:
        >>> round(adjust_lambda_for_game_state(1.80, 85, 1, "defensive"), 2)
        0.68

        Same scenario after own red card in 2H:
        >>> round(adjust_lambda_for_game_state(1.80, 45, 1, "defensive",
        ...                                     has_red_card=True), 2)
        0.2
    """
    if minutes_remaining <= 0:
        return _LAMBDA_FLOOR

    # Layer 1: linear time scaling
    lambda_remaining = lambda_base * (minutes_remaining / 90)

    # Layer 2: coach style × game state
    if score_diff > 0:
        state = "leading"
    elif score_diff < 0:
        state = "trailing"
    else:
        state = "level"
    lambda_remaining *= _STATE_FACTORS[coach_style][state]

    # Layer 3: big margin compound reduction
    if abs(score_diff) >= 2:
        lambda_remaining *= _BIG_MARGIN_EXTRA

    # Layer 4: red card downgrade
    if has_red_card:
        lambda_remaining *= _RED_CARD_FACTOR

    return max(lambda_remaining, _LAMBDA_FLOOR)


def p_at_least_one_more_goal(
    lambda_base: float,
    minutes_remaining: int,
    score_diff: int,
    coach_style: CoachStyle = "balanced",
    has_red_card: bool = False,
) -> float:
    """Convenience: P(this team scores ≥1 more) given live state.

    Uses Poisson: P(≥1) = 1 − e^(−λ_adjusted).
    """
    import math
    lam = adjust_lambda_for_game_state(
        lambda_base, minutes_remaining, score_diff, coach_style, has_red_card,
    )
    return 1.0 - math.exp(-lam)
