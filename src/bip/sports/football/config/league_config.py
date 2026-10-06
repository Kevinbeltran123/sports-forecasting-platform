"""Pydantic models for league configuration."""

from pydantic import BaseModel, field_validator


class ApiMappings(BaseModel):
    """External API identifier mappings for a league."""

    api_football_league_id: int
    odds_api_sport_key: str
    team_name_mappings: dict[str, str] = {}


class SeasonStructure(BaseModel):
    """Season calendar and structure for a league."""

    start_month: int
    end_month: int
    typical_matchdays: int
    winter_break: bool = False

    @field_validator("start_month", "end_month")
    @classmethod
    def validate_month(cls, v: int) -> int:
        if not 1 <= v <= 12:
            raise ValueError(f"Month must be between 1 and 12, got {v}")
        return v


class ModelParams(BaseModel):
    """Model configuration parameters per league."""

    calibration_method: str = "isotonic"
    edge_threshold_btts: float = 0.05
    edge_threshold_ah: float = 0.06
    edge_threshold_ou: float = 0.05
    edge_threshold_1x2: float = 0.08
    edge_threshold_corners: float = 0.07
    min_sample_size: int = 500

    @field_validator("calibration_method")
    @classmethod
    def validate_calibration_method(cls, v: str) -> str:
        allowed = {"isotonic", "platt"}
        if v not in allowed:
            raise ValueError(f"calibration_method must be one of {allowed}, got '{v}'")
        return v

    @field_validator(
        "edge_threshold_btts",
        "edge_threshold_ah",
        "edge_threshold_ou",
        "edge_threshold_1x2",
        "edge_threshold_corners",
    )
    @classmethod
    def validate_edge_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Edge threshold must be between 0.0 and 1.0, got {v}")
        return v


class LeagueConfig(BaseModel):
    """Complete configuration for a single league."""

    name: str
    slug: str
    country: str
    api_mappings: ApiMappings
    season_structure: SeasonStructure
    model_params: ModelParams = ModelParams()
