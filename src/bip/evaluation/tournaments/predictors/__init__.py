"""Match prediction models — three-way comparison for goals + count markets.

Renamed from `models/` to `predictors/` to avoid collision with the
sibling `models.py` (Pydantic entity models). The two are distinct:
- `models.py`     — data shapes (Tournament, Player, Squad, etc.)
- `predictors/`   — prediction algorithms (Poisson variants, ELO, ...)
"""
