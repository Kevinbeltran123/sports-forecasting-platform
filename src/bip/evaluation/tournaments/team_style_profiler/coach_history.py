"""Current head coach for WC2026-relevant national teams.

DATA SOURCE: public Transfermarkt + FIFA + federation records (May 2026).
Verified manually. Update before WC2026 kicks off (June 2026) — a DT
change in May/June would require refresh.

Used by TSP to filter fixtures to the current-coach spell ONLY. Any
fixture before ``start_date`` is excluded from the team's TSV.

Lookup is by canonical English team name. API-Football team IDs are
included for downstream pulls.

OPERATIONAL RULE (per design):
  - If <10 matches with current coach since 2023-01-01 -> flag = 'red'.
  - If 5-9 matches -> flag = 'yellow' (low confidence).
  - If >=10 matches -> flag = 'green'.

When the current coach took over BEFORE 2023-01-01, the effective cutoff
is 2023-01-01 (we only look at the post-WC22 cycle regardless).
When the current coach took over AFTER 2023-01-01, the effective cutoff
is the coach's start date.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class CoachRecord:
    team_name: str
    api_football_team_id: int
    coach_name: str
    coach_start_date: date
    """Date of first match in the current spell."""
    confederation: str
    """One of UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC."""
    notes: str = ""


# --- WC2026 qualified or strong-qualifying teams as of May 2026 ---
# Coach data verified manually. Update if any DT changes before kick-off.
CURRENT_COACHES: dict[str, CoachRecord] = {
    # === UEFA ===
    "France": CoachRecord("France", 2, "Didier Deschamps", date(2012, 7, 8), "UEFA",
        notes="Multi-cycle, very stable; ends after WC2026."),
    "Germany": CoachRecord("Germany", 25, "Julian Nagelsmann", date(2023, 9, 22), "UEFA",
        notes="Replaced Flick post-WC22 disaster."),
    "England": CoachRecord("England", 10, "Thomas Tuchel", date(2025, 1, 1), "UEFA",
        notes="Replaced Southgate post-Euro24. NEW COACH — likely red/yellow flag at WC."),
    "Spain": CoachRecord("Spain", 9, "Luis de la Fuente", date(2022, 12, 9), "UEFA"),
    "Italy": CoachRecord("Italy", 768, "Luciano Spalletti", date(2023, 8, 18), "UEFA",
        notes="Italy may not qualify for WC26 (playoff route)."),
    "Portugal": CoachRecord("Portugal", 27, "Roberto Martinez", date(2023, 1, 9), "UEFA"),
    "Netherlands": CoachRecord("Netherlands", 1118, "Ronald Koeman", date(2023, 1, 1), "UEFA",
        notes="Second spell — first was 2018-2020."),
    "Belgium": CoachRecord("Belgium", 1, "Rudi Garcia", date(2025, 1, 28), "UEFA",
        notes="Replaced Tedesco. NEW COACH — likely red/yellow."),
    "Croatia": CoachRecord("Croatia", 3, "Zlatko Dalic", date(2017, 10, 7), "UEFA",
        notes="Long tenure, stable."),
    "Switzerland": CoachRecord("Switzerland", 15, "Murat Yakin", date(2021, 8, 1), "UEFA"),
    "Denmark": CoachRecord("Denmark", 21, "Brian Riemer", date(2025, 6, 1), "UEFA",
        notes="Recently appointed. RED FLAG until n=10."),
    "Poland": CoachRecord("Poland", 24, "Michal Probierz", date(2023, 9, 20), "UEFA"),
    "Austria": CoachRecord("Austria", 4, "Ralf Rangnick", date(2022, 5, 1), "UEFA"),
    "Turkey": CoachRecord("Turkey", 19, "Vincenzo Montella", date(2023, 9, 14), "UEFA"),
    "Norway": CoachRecord("Norway", 5, "Stale Solbakken", date(2020, 12, 7), "UEFA"),
    "Serbia": CoachRecord("Serbia", 30, "Veljko Paunovic", date(2025, 1, 1), "UEFA",
        notes="Replaced Stojkovic. NEW COACH."),
    "Scotland": CoachRecord("Scotland", 1108, "Steve Clarke", date(2019, 5, 20), "UEFA"),

    # === CONMEBOL ===
    "Argentina": CoachRecord("Argentina", 26, "Lionel Scaloni", date(2018, 8, 23), "CONMEBOL",
        notes="Won WC22 + Copa21+24. Long stable spell — high-quality TSV."),
    "Brazil": CoachRecord("Brazil", 6, "Carlo Ancelotti", date(2025, 5, 26), "CONMEBOL",
        notes="VERY NEW. RED FLAG. First non-Brazilian coach in modern era."),
    "Uruguay": CoachRecord("Uruguay", 7, "Marcelo Bielsa", date(2023, 5, 15), "CONMEBOL"),
    "Colombia": CoachRecord("Colombia", 8, "Nestor Lorenzo", date(2022, 6, 1), "CONMEBOL"),
    "Ecuador": CoachRecord("Ecuador", 1569, "Sebastian Beccacece", date(2024, 8, 1), "CONMEBOL"),
    "Paraguay": CoachRecord("Paraguay", 12, "Gustavo Alfaro", date(2024, 8, 21), "CONMEBOL",
        notes="Took over from Pintado. yellow/red until n=10."),

    # === CAF ===
    "Morocco": CoachRecord("Morocco", 31, "Mohamed Ouahbi", date(2026, 3, 5), "CAF",
        notes="Replaced Regragui 2026-03-05 (after AFCON25 final loss). U20 WC25 winner, "
              "no prior senior spell. RED FLAG <3mo pre-WC — WC22/AFCON legacy NOT predictive."),
    "Senegal": CoachRecord("Senegal", 1525, "Pape Thiaw", date(2025, 2, 5), "CAF",
        notes="Replaced Aliou Cisse. NEW COACH — yellow/red flag."),
    "Egypt": CoachRecord("Egypt", 1528, "Hossam Hassan", date(2024, 4, 26), "CAF"),
    "Algeria": CoachRecord("Algeria", 1546, "Vladimir Petkovic", date(2024, 8, 14), "CAF"),
    "Tunisia": CoachRecord("Tunisia", 1546, "Sami Trabelsi", date(2024, 12, 13), "CAF"),
    "Ghana": CoachRecord("Ghana", 1556, "Otto Addo", date(2024, 2, 22), "CAF",
        notes="Second spell, lost AFCON 2023 group stage."),
    "Nigeria": CoachRecord("Nigeria", 1543, "Eric Chelle", date(2025, 1, 8), "CAF",
        notes="NEW. Took over from Peseiro."),
    "Côte d'Ivoire": CoachRecord("Côte d'Ivoire", 1525, "Emerse Fae", date(2024, 1, 24), "CAF",
        notes="Won AFCON 2023 after promotion mid-tournament."),
    "Cameroon": CoachRecord("Cameroon", 1542, "Marc Brys", date(2024, 4, 2), "CAF"),
    "South Africa": CoachRecord("South Africa", 1549, "Hugo Broos", date(2021, 5, 5), "CAF",
        notes="Long tenure, AFCON 2023 semifinalist — good sample for TSV."),
    "Mali": CoachRecord("Mali", 1539, "Tom Saintfiet", date(2024, 12, 1), "CAF"),
    "Cape Verde": CoachRecord("Cape Verde", 1554, "Pedro Leitao Brito 'Bubista'", date(2020, 3, 1), "CAF",
        notes="Long tenure; AFCON 2023 QF over-performer."),

    # === AFC ===
    "Japan": CoachRecord("Japan", 13, "Hajime Moriyasu", date(2018, 7, 26), "AFC",
        notes="Long tenure including WC22."),
    "South Korea": CoachRecord("South Korea", 18, "Hong Myung-bo", date(2024, 7, 8), "AFC",
        notes="Took over from Klinsmann mid-cycle."),
    "Australia": CoachRecord("Australia", 20, "Tony Popovic", date(2024, 9, 23), "AFC",
        notes="NEW. Took over from Arnold."),
    "Iran": CoachRecord("Iran", 22, "Amir Ghalenoei", date(2023, 3, 1), "AFC"),
    "Saudi Arabia": CoachRecord("Saudi Arabia", 32, "Herve Renard", date(2024, 12, 1), "AFC",
        notes="Returned for second spell."),
    "Uzbekistan": CoachRecord("Uzbekistan", 1547, "Timur Kapadze", date(2024, 4, 1), "AFC",
        notes="First WC26 qualification ever for Uzbekistan."),
    "Jordan": CoachRecord("Jordan", 1568, "Jamal Sellami", date(2024, 1, 1), "AFC",
        notes="First WC qualification."),

    # === CONCACAF (incl. WC2026 hosts) ===
    "USA": CoachRecord("USA", 2384, "Mauricio Pochettino", date(2024, 10, 1), "CONCACAF",
        notes="HOST nation. NEW COACH (took over Sep 2024). Combined "
              "host + new-coach + CONCACAF blind spots — strongest fade signal."),
    "Mexico": CoachRecord("Mexico", 16, "Javier Aguirre", date(2024, 8, 6), "CONCACAF",
        notes="HOST nation. Took over from Lozano. yellow until n=10."),
    "Canada": CoachRecord("Canada", 1118, "Jesse Marsch", date(2024, 5, 13), "CONCACAF",
        notes="HOST nation. NEW COACH from May 2024."),
    "Costa Rica": CoachRecord("Costa Rica", 11, "Miguel Herrera", date(2025, 1, 14), "CONCACAF"),
    "Panama": CoachRecord("Panama", 17, "Thomas Christiansen", date(2020, 7, 1), "CONCACAF",
        notes="Long tenure."),
    "Jamaica": CoachRecord("Jamaica", 1532, "Steve McClaren", date(2024, 7, 1), "CONCACAF"),

    # === OFC ===
    "New Zealand": CoachRecord("New Zealand", 1109, "Darren Bazeley", date(2023, 12, 7), "OFC"),

    # === WC2026 PLAYOFF QUALIFIERS (added 2026-05-28) ===
    # UEFA
    "Bosnia and Herzegovina": CoachRecord("Bosnia and Herzegovina", 1113, "Sergej Barbarez", date(2024, 4, 1), "UEFA",
        notes="Appointed 2024 — yellow until n=10."),
    "Czech Republic": CoachRecord("Czech Republic", 770, "Ivan Hašek", date(2024, 3, 1), "UEFA",
        notes="Second spell as Czech NT head coach."),
    "Sweden": CoachRecord("Sweden", 5, "Jon Dahl Tomasson", date(2024, 2, 1), "UEFA",
        notes="Former Blackburn Rovers manager."),
    # CAF
    "DR Congo": CoachRecord("DR Congo", 1508, "Sébastien Desabre", date(2023, 3, 1), "CAF",
        notes="AFCON 2023 semifinalist — solid sample for TSV."),
    # AFC
    "Iraq": CoachRecord("Iraq", 1567, "Graham Arnold", date(2025, 5, 1), "AFC",
        notes="NEW — ex-Australia coach. RED FLAG until n=10."),
    "Qatar": CoachRecord("Qatar", 1569, "Julen Lopetegui", date(2025, 5, 1), "AFC",
        notes="VERY NEW. RED FLAG. Spain/Wolves/AC Milan background."),
    # CONCACAF
    "Curaçao": CoachRecord("Curaçao", 5530, "Dick Advocaat", date(2024, 1, 1), "CONCACAF",
        notes="Veteran Dutch coach. Curaçao has limited international fixture history."),
    "Haiti": CoachRecord("Haiti", 2386, "Sébastien Migné", date(2024, 3, 1), "CONCACAF",
        notes="Appointed early 2024."),
}


def get_current_coach(team_name: str) -> CoachRecord | None:
    """Return the current head coach record for a team.

    Lookup is case-sensitive on the canonical English team name. Returns
    None when the team is not in the WC2026-relevant set.

    >>> r = get_current_coach("Argentina")
    >>> r.coach_name
    'Lionel Scaloni'
    >>> r.start_date.year
    2018
    """
    return CURRENT_COACHES.get(team_name)


def get_effective_filter_date(team_name: str, era_start: date = date(2023, 1, 1)) -> date | None:
    """Return the effective cutoff date for filtering this team's fixtures.

    The cutoff is ``max(coach_start_date, era_start)``. Any fixture
    BEFORE this date is excluded from the team's TSV.

    Returns None when the team isn't in the registry.

    >>> from datetime import date
    >>> get_effective_filter_date("Argentina")
    datetime.date(2023, 1, 1)
    >>> # Argentina's Scaloni started 2018, but era_start=2023-01-01 wins.
    >>> get_effective_filter_date("Brazil")
    datetime.date(2025, 5, 26)
    >>> # Brazil's Ancelotti started May 2025, post-era-start.
    """
    record = get_current_coach(team_name)
    if record is None:
        return None
    return max(record.coach_start_date, era_start)


def is_new_coach(team_name: str, threshold_days: int = 365, today: date | None = None) -> bool:
    """True iff the current coach took over within the last ``threshold_days``.

    Used to surface a "NEW COACH" warning: teams with a recent coach
    change are higher-uncertainty because we have fewer matches to
    profile.

    >>> import datetime
    >>> # Brazil (Ancelotti started 2025-05-26) is a new coach as of 2026-05-24
    >>> is_new_coach("Brazil", today=datetime.date(2026, 5, 24))
    True
    >>> # Argentina (Scaloni since 2018) is NOT a new coach
    >>> is_new_coach("Argentina", today=datetime.date(2026, 5, 24))
    False
    """
    record = get_current_coach(team_name)
    if record is None:
        return False
    if today is None:
        today = date.today()
    return (today - record.coach_start_date).days < threshold_days


def teams_by_confederation(conf: str) -> list[str]:
    """Return the list of WC2026-relevant team names for a confederation.

    >>> ueafa = teams_by_confederation("CONMEBOL")
    >>> "Argentina" in ueafa
    True
    >>> "France" in ueafa
    False
    """
    return sorted(
        name for name, r in CURRENT_COACHES.items() if r.confederation == conf
    )
