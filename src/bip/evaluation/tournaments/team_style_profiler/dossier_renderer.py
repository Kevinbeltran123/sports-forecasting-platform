"""Render MatchDossier to Markdown and JSON formats.

Per operator decision: both formats emitted side-by-side.
"""
from __future__ import annotations

import json
from typing import Any

from bip.evaluation.tournaments.team_style_profiler.match_dossier import (
    MatchDossier,
    Pick,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import (
    TeamStyleVector,
)


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _render_tsv_block(tsv: TeamStyleVector | None, name: str) -> list[str]:
    if tsv is None:
        return [f"### {name}", "", "*Profile not available — cohort fallback used.*", ""]
    out = [
        f"### {name} — {tsv.coach.coach_name} (since {tsv.coach.start_date.isoformat()})",
        "",
        f"- **Profile flag:** `{tsv.flag}` (n={tsv.n_matches} matches)",
        f"- **Confederation:** {tsv.confederation}",
        "",
        "**Offensive:**",
        f"- Goals for: **{tsv.goals_for_per_match.mean:.2f}**/match "
        f"[{tsv.goals_for_per_match.ci_low:.2f}, {tsv.goals_for_per_match.ci_high:.2f}]",
        f"- Shots: {tsv.shots_per_match.mean:.1f}, on target: "
        f"{tsv.shots_on_target_per_match.mean:.1f} "
        f"({_fmt_pct(tsv.shots_on_target_ratio.mean)} ratio)",
        f"- Corners for: {tsv.corners_for_per_match.mean:.1f}/match",
        f"- Possession: {tsv.possession_avg.mean:.1f}%",
        "",
        "**Defensive:**",
        f"- Goals against: **{tsv.goals_against_per_match.mean:.2f}**/match "
        f"[{tsv.goals_against_per_match.ci_low:.2f}, {tsv.goals_against_per_match.ci_high:.2f}]",
        f"- Clean sheets: {_fmt_pct(tsv.clean_sheet_rate.mean)}",
        f"- Corners against: {tsv.corners_against_per_match.mean:.1f}/match",
        f"- Offsides forced: {tsv.offsides_against_per_match.mean:.1f}/match (proxy high line)",
        "",
        "**Character of match:**",
        f"- Yellow cards: {tsv.yellow_cards_per_match.mean:.2f}/match",
        f"- Fouls: {tsv.fouls_per_match.mean:.1f}/match",
        f"- BTTS rate: {_fmt_pct(tsv.btts_rate.mean)}",
        f"- O/U 2.5 rate: O{_fmt_pct(tsv.over_25_rate.mean)} / "
        f"U{_fmt_pct(1 - tsv.over_25_rate.mean)}",
        f"- Mean total goals: {tsv.mean_total_goals.mean:.2f}",
        "",
        "**Goals distribution by 15-min bucket:**",
    ]
    g15 = tsv.goals_for_per_15min
    for label, b in [
        ("0-14", g15.bucket_0_14), ("15-29", g15.bucket_15_29),
        ("30-44", g15.bucket_30_44), ("45-59", g15.bucket_45_59),
        ("60-74", g15.bucket_60_74), ("75-90", g15.bucket_75_90),
    ]:
        bar_len = int(b.mean * 50)
        bar = "█" * bar_len
        out.append(f"  - `{label}min` {b.mean:.3f} {bar}")
    out.append("")

    if tsv.sub_profiles:
        out.append(f"**Sub-profiles vs confederation:**")
        for conf, sp in sorted(tsv.sub_profiles.items()):
            out.append(
                f"  - vs **{conf}** (n={sp.n_matches_vs_conf}): "
                f"GF {sp.goals_for_per_match.mean:.2f}, "
                f"GA {sp.goals_against_per_match.mean:.2f}, "
                f"BTTS {_fmt_pct(sp.btts_rate.mean)}, "
                f"O2.5 {_fmt_pct(sp.over_25_rate.mean)}"
            )
        out.append("")

    return out


def _render_identity_block(tid: Any, name: str) -> list[str]:
    if tid is None:
        return [f"- **{name}:** *sin identidad táctica (sin cobertura StatsBomb ni scouting)*"]
    if tid.confidence == "scouting":
        dims = (
            f"  - presión: `{tid.press_intensity}` · balón parado: `{tid.set_piece_reliance}` · "
            f"definición: `{tid.finishing_profile}` *(scouting, sin métricas)*"
        )
    else:
        dims = (
            f"  - presión: `{tid.press_intensity}` (PPDA {tid.ppda:.1f}) · "
            f"balón parado: `{tid.set_piece_reliance}` ({tid.set_piece_xg_share:.0%} xG) · "
            f"definición: `{tid.finishing_profile}` (conv {tid.conversion_rate:.2f})"
        )
    out = [
        f"- **{name}: {tid.archetype}** (`{tid.confidence}`"
        + (f", n={tid.n_matches}" if tid.confidence != "scouting" else "") + ")",
        dims,
    ]
    if tid.anchor is not None:
        out.append(f"  - gatillo de presión: {tid.anchor.press_trigger}")
        out.append(f"  - construcción: {tid.anchor.build_up}")
        out.append(f"  - eslabón débil: {tid.anchor.weak_link}")
    return out


def _render_matchup_read(dossier: MatchDossier) -> list[str]:
    r = dossier.matchup_read
    if r is None:
        return []
    ctx = dossier.context
    out = [
        "---",
        "",
        "## §1b. Lectura de planteamiento (game-plan)",
        "",
    ]
    out.extend(_render_identity_block(r.home, ctx.home_team))
    out.extend(_render_identity_block(r.away, ctx.away_team))
    out.extend([
        "",
        f"- **Tempo:** {r.tempo}",
        f"- **Forma del partido:** {r.game_shape}",
        "",
        "**Hacia qué mercado empuja el cruce:**",
    ])
    if r.market_leans:
        for lean in r.market_leans:
            out.append(f"- {lean.text}")
    else:
        out.append("- *(sin lean táctico claro — planteamientos flexibles)*")
    if r.caveats:
        out.append("")
        out.append("**Cautelas:**")
        for c in r.caveats:
            out.append(f"- ⚠ {c}")
    out.append("")
    return out


def _format_transfer_line(team: str, transfer_dict: dict) -> str:
    if not transfer_dict["has_signal"]:
        return f"- **{team}:** no signal"
    tilt = transfer_dict["tilt"]
    evidence = transfer_dict["evidence"]
    return f"- **{team}:** ×{tilt:.2f} — {evidence}"


def _render_pick(pick: Pick, idx: int) -> list[str]:
    # Evidence, not verdict: no STRONG/MODERATE category, no numeric score
    # (operator decides — per the analyst reframe). Ordering is preserved by the
    # caller's internal sort; we just don't surface a false-precision number.
    out = [
        f"### {idx}. `{pick.direction}` {pick.market}",
        "",
        "**Por qué (evidencia):**",
    ]
    for r in pick.rationale:
        out.append(f"- {r}")
    if pick.risk_flags:
        out.append("")
        out.append("**Banderas de riesgo:**")
        for f in pick.risk_flags:
            out.append(f"- ⚠ {f}")
    out.append("")
    return out


def render_markdown(dossier: MatchDossier) -> str:
    ctx = dossier.context
    out = [
        f"# Match Dossier — {ctx.home_team} vs {ctx.away_team}",
        "",
        f"**Tournament:** `{ctx.tournament_slug}`  ",
        f"**Date:** {ctx.fixture_date.isoformat()}  ",
        f"**Generated:** {dossier.generated_at.isoformat(timespec='seconds')}",
        "",
        "---",
        "",
        "## §1. Team Fundamentals",
        "",
    ]
    out.extend(_render_tsv_block(ctx.home_tsv, ctx.home_team))
    out.extend(_render_tsv_block(ctx.away_tsv, ctx.away_team))

    out.extend(_render_matchup_read(dossier))

    out.extend([
        "---",
        "",
        "## §2. Applicable Patterns (Tendencies)",
        "",
        "### Host nation filter (L5.6)",
        "",
        f"- **Host in fixture?** `{dossier.host_verdict['host_in_fixture']}`",
        f"- **Downgrade factor:** ×{dossier.host_verdict['downgrade_factor']}",
        f"- {dossier.host_verdict['rationale']}",
        "",
        "### Confederation tilt (Iter 3 + 4)",
        "",
        f"- Home `{dossier.conf_tilt['home_conf']}` × Away `{dossier.conf_tilt['away_conf']}` "
        f"(cross-conf: {dossier.conf_tilt['is_cross_conf']})",
        f"- Home tilt: ×{dossier.conf_tilt['home_tilt']:.2f} | "
        f"Away tilt: ×{dossier.conf_tilt['away_tilt']:.2f}",
        f"- Edge-threshold multiplier (predictor reliability): "
        f"×{dossier.conf_tilt['edge_threshold_multiplier']:.2f}",
        f"- {dossier.conf_tilt['rationale']}",
        "",
        "### Team transfer signals (Iter 2)",
        "",
        _format_transfer_line(ctx.home_team, dossier.home_team_transfer),
        _format_transfer_line(ctx.away_team, dossier.away_team_transfer),
        "",
    ])

    if dossier.regime_warning:
        out.extend([
            "### Regime warning (L3.1)",
            "",
            f"- **Modern era:** {dossier.regime_warning['is_modern_era']}",
            f"- {dossier.regime_warning['rationale']}",
            "",
        ])

    out.extend([
        "---",
        "",
        "## §3. Señales de mercado (evidencia — el operador decide, sin veredicto)",
        "",
    ])

    if not dossier.picks:
        out.extend([
            "*No picks emerged from current rules. Suggests:* both teams generic, "
            "no strong patterns active, OR sample too small for confidence.",
            "",
        ])
    else:
        # Group risk flags shown only once at top of picks
        risk_flags_set: set[str] = set()
        for p in dossier.picks:
            risk_flags_set.update(p.risk_flags)
        if risk_flags_set:
            out.append("**Risk flags applying to this fixture:**")
            for f in sorted(risk_flags_set):
                out.append(f"- ⚠ {f}")
            out.append("")

        for i, pick in enumerate(dossier.picks, 1):
            out.extend(_render_pick(pick, i))

    out.extend([
        "---",
        "",
        f"*Generated by TSP Match Dossier — TSV + patterns_v2 synthesis.*",
    ])

    return "\n".join(out)


def render_json(dossier: MatchDossier) -> str:
    """Structured JSON for downstream pipelines."""
    def tsv_to_dict(tsv: TeamStyleVector | None) -> Any:
        return tsv.model_dump(mode="json") if tsv else None

    def tid_to_dict(tid: Any) -> Any:
        if tid is None:
            return None
        return {
            "archetype": tid.archetype,
            "confidence": tid.confidence,
            "n_matches": tid.n_matches,
            "press_intensity": tid.press_intensity,
            "set_piece_reliance": tid.set_piece_reliance,
            "finishing_profile": tid.finishing_profile,
            "ppda": round(tid.ppda, 2) if tid.ppda is not None else None,
            "set_piece_xg_share": round(tid.set_piece_xg_share, 3) if tid.set_piece_xg_share is not None else None,
            "conversion_rate": round(tid.conversion_rate, 3) if tid.conversion_rate is not None else None,
        }

    ctx = dossier.context
    r = dossier.matchup_read
    payload = {
        "context": {
            "home_team": ctx.home_team,
            "away_team": ctx.away_team,
            "tournament_slug": ctx.tournament_slug,
            "fixture_date": ctx.fixture_date.isoformat(),
            "home_tsv": tsv_to_dict(ctx.home_tsv),
            "away_tsv": tsv_to_dict(ctx.away_tsv),
        },
        "game_plan": None if r is None else {
            "home_identity": tid_to_dict(r.home),
            "away_identity": tid_to_dict(r.away),
            "tempo": r.tempo,
            "game_shape": r.game_shape,
            "market_leans": [
                {"market": l.market, "direction": l.direction, "rationale": l.rationale}
                for l in r.market_leans
            ],
            "caveats": r.caveats,
        },
        "patterns": {
            "host_verdict": dossier.host_verdict,
            "conf_tilt": dossier.conf_tilt,
            "home_team_transfer": dossier.home_team_transfer,
            "away_team_transfer": dossier.away_team_transfer,
            "regime_warning": dossier.regime_warning,
        },
        "picks": [
            {
                "market": p.market,
                "direction": p.direction,
                "category": p.category,
                "score": p.score,
                "rationale": p.rationale,
                "risk_flags": p.risk_flags,
            }
            for p in dossier.picks
        ],
        "generated_at": dossier.generated_at.isoformat(),
    }
    return json.dumps(payload, indent=2)
