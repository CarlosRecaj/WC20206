# =============================================================================
# WC2026 — Preparació del Dataset d'Equips Participants
# =============================================================================
# Aquest script construeix el fitxer FIFA_2026_Data.csv que alimenta
# la simulació Montecarlo. Combina tres fonts:
#   1. partits_a_jugar.csv  → llista d'equips participants al torneig
#   2. all_matches.csv      → historial de partits per calcular PPM i DGPM
#   3. fifa_ranking_2026-06-08.csv → punts FIFA actuals de cada equip
# =============================================================================

import pandas as pd
import numpy as np


# =============================================================================
# CONSTANTS GLOBALS
# =============================================================================
# Centralitzem aquí els paràmetres que podrien canviar entre edicions
# del torneig o actualitzacions de dades.

# Equips que juguen en terreny propi: reben un avantatge de local
# perquè estan acostumats als estadis, el públic els empeny i no fan viatge llarg.
HOST_NATIONS = {"Canada", "Mexico", "USA"}

# Tots els partits de la fase de grups tenen match_id < 73 (format WC 2026 amb 48 equips)
GROUP_STAGE_MAX_ID = 73

# Volem les estadístiques del cicle classificatori que desemboca al WC2026
TARGET_CYCLE = 2026

# Fitxers d'entrada i sortida
MATCHES_CSV      = "partits_a_jugar.csv"
ALL_MATCHES_CSV  = "all_matches.csv"
RANKING_CSV      = "fifa_ranking_2026-06-08.csv"
OUTPUT_CSV       = "FIFA_2026_Data.csv"

# Columnes del rànquing FIFA que no necessitem (redundants o no usades pel model)
RANKING_COLS_TO_DROP = [
    "team_code", "association", "rank",
    "previous_rank", "previous_points", "rated_matches",
]

# Diccionari de correccions de noms: el CSV de partits i el de rànquings
# no sempre coincideixen en el nom oficial de cada país.
# Quan hi ha discrepàncies, el merge no troba l'equip i genera NaN als punts FIFA.
NOM_CORRECTIONS = {
    "Brunei":                  "Brunei Darussalam",
    "Cape Verde":              "Cabo Verde",
    "China":                   "China PR",
    "DR Congo":                "Congo DR",
    "East Timor":              "Timor-Leste",
    "Eastern Samoa":           "American Samoa",
    "Gambia":                  "The Gambia",
    "Iran":                    "IR Iran",
    "Ireland":                 "Republic of Ireland",
    "Ivory Coast":             "Côte d'Ivoire",
    "Kyrgyzstan":              "Kyrgyz Republic",
    "Macao":                   "Macau",
    "Macedonia":               "North Macedonia",
    "North Korea":             "Korea DPR",
    "Saint Kitts":             "St. Kitts and Nevis",
    "Saint Kitts and Nevis":   "St. Kitts and Nevis",
    "Saint Lucia":             "St. Lucia",
    "South Korea":             "Korea Republic",
    "St Vincent & Grenadines": "St. Vincent and the Grenadines",
    "Swaziland":               "Eswatini",
    "São Tome and Principe":   "São Tomé and Príncipe",
    "Taiwan":                  "Chinese Taipei",
    "United States":           "USA",
    "Western Samoa":           "Samoa",
    "Turkey":                  "Türkiye",
}


# =============================================================================
# 1. LLISTA D'EQUIPS PARTICIPANTS
# =============================================================================
# Extraiem els 48 equips directament del CSV de partits de fase de grups.
# Usem match_id < GROUP_STAGE_MAX_ID perquè els partits eliminatoris
# no contenen noms d'equips reals sinó referències com "Winner match 50".

df_matches = pd.read_csv(MATCHES_CSV)
group_matches = df_matches[df_matches["match_id"] < GROUP_STAGE_MAX_ID]

# Construïm la llista unificant team1 i team2 i eliminant duplicats
participating_teams = sorted(
    set(group_matches["team1"].tolist() + group_matches["team2"].tolist())
)

# Apliquem les correccions de noms per alinear amb el rànquing FIFA
participating_teams = [NOM_CORRECTIONS.get(t, t) for t in participating_teams]

print(f"Equips participants ({len(participating_teams)}): {participating_teams}")


# =============================================================================
# 2. RÀNQUING FIFA ACTUAL
# =============================================================================
# Llegim el rànquing de juny del 2026 (el més proper a l'inici del torneig)
# i ens quedem només amb nom i punts FIFA de cada selecció.

df_ranking = (
    pd.read_csv(RANKING_CSV)
    .drop(columns=RANKING_COLS_TO_DROP)
    .assign(team=lambda x: x["team"].replace(NOM_CORRECTIONS))
)


# =============================================================================
# 3. ESTADÍSTIQUES DE FORMA DEL CICLE 2022–2026
# =============================================================================
# El rànquing FIFA captura la força global d'un equip, però no la seva forma
# recent. Per complementar-lo, calculem punts per partit (ppm) i diferència
# de gols per partit (dgpm) en partits oficials del cicle classificatori 2026.
# Excloem amistosos (massa poc representatius) i mundials (no pertanyen al cicle).

# --- 3a. Funció d'assignació de cicle mundialista ---
def assign_world_cup_cycle(year: int) -> int:
    """Retorna l'any del pròxim mundial per a un any donat.

    Els mundials masculins cauen en anys on (any % 4) == 2 (ex: 2022, 2026).
    Fem servir un diccionari d'offsets per evitar el if/elif encadenat.
    """
    offsets = {3: 3, 0: 2, 1: 1, 2: 0}
    return year + offsets[year % 4]


# --- 3b. Filtrat i preparació ---
df_qualifying = (
    pd.read_csv(ALL_MATCHES_CSV)
    .query("tournament != 'Friendly' and tournament != 'World Cup'")
    .drop(columns=["country"])
    .assign(
        home_team=lambda x: x["home_team"].replace(NOM_CORRECTIONS),
        away_team=lambda x: x["away_team"].replace(NOM_CORRECTIONS),
        # errors='coerce' converteix dates malformades a NaT en lloc de trencar
        date=lambda x: pd.to_datetime(x["date"], errors="coerce"),
    )
    .dropna(subset=["date"])
    .assign(year=lambda x: x["date"].dt.year.astype(int))
    .assign(wc_cycle=lambda x: x["year"].apply(assign_world_cup_cycle))
    # Filtrem NOMÉS els equips participants per no calcular estadístiques
    # d'equips que no juguen el torneig (redueix el temps de còmput)
    .loc[lambda x:
        x["home_team"].isin(participating_teams) |
        x["away_team"].isin(participating_teams)
    ]
    .loc[lambda x: x["wc_cycle"] == TARGET_CYCLE]
)

# --- 3c. Càlcul de punts i diferència de gols ---
# Reutilitzem la mateixa funció per a la perspectiva local i visitant
# per evitar duplicar la lògica (DRY).

def compute_team_stats(df: pd.DataFrame, team_col: str, score_col: str, opp_col: str) -> pd.DataFrame:
    """Calcula punts i diferència de gols des de la perspectiva d'un equip."""
    return (
        df[["date", team_col, score_col, opp_col, "wc_cycle"]]
        .rename(columns={team_col: "team"})
        .assign(
            points=lambda x: np.where(
                x[score_col] > x[opp_col], 3,
                np.where(x[score_col] == x[opp_col], 1, 0),
            ),
            goal_diff=lambda x: x[score_col] - x[opp_col],
        )
    )


home_stats = compute_team_stats(df_qualifying, "home_team", "home_score", "away_score")
away_stats = compute_team_stats(df_qualifying, "away_team", "away_score", "home_score")

df_cycle_stats = (
    pd.concat([home_stats, away_stats])
    .groupby(["team", "wc_cycle"])
    .agg(
        matches_played=("points", "count"),
        total_points=("points", "sum"),
        total_goal_diff=("goal_diff", "sum"),
    )
    .assign(
        points_per_match=lambda x: (x["total_points"] / x["matches_played"]).round(2),
        dif_goals_per_match=lambda x: (x["total_goal_diff"] / x["matches_played"]).round(2),
        # Marquem els equips locals perquè la simulació els pugui afegir l'avantatge de camp
        home_advantage=lambda x: x.index.get_level_values("team").isin(HOST_NATIONS).astype(int),
    )
    .reset_index()
    # Ens quedem només amb els equips que realment juguen el torneig
    .loc[lambda x: x["team"].isin(participating_teams)]
)


# =============================================================================
# 4. COMBINACIÓ FINAL I EXPORTACIÓ
# =============================================================================
# Unim les estadístiques del cicle amb els punts FIFA actuals.
# Usem how='left' per conservar tots els equips del cicle encara que
# algun no aparegués al rànquing FIFA (cas poc probable però possible).

df_final = (
    df_cycle_stats
    .merge(df_ranking, on="team", how="left")
    # Reordenem per llegibilitat: primer identitat, després mètriques
    [["team", "wc_cycle", "matches_played", "total_points", "total_goal_diff",
      "points_per_match", "dif_goals_per_match", "home_advantage", "points"]]
)

df_final.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Dataset exportat a '{OUTPUT_CSV}' amb {len(df_final)} equips.")
print(df_final.head())
