# =============================================================================
# WC2026 — Simulació Montecarlo amb Regressió Logística
# =============================================================================
# Versió revisada i millorada: imports agrupats, constants al capdamunt,
# funcions ben documentades i lògica interna netejada.
# =============================================================================

import random
from collections import Counter

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS GLOBALS
# =============================================================================
# Centralitzem aquí tot el que podria canviar entre execucions:
# noms de fitxers, ordre de features, ID del partit final, etc.
# Així no cal anar a buscar "màgia enterrada" dins de les funcions.

FEATURE_ORDER = [
    "points_diff",
    "points_sum",
    "ppm_diff",
    "gdpm_diff",
    "same_confederation",
    "home_advantage",
]

# ID del partit de la gran final al CSV de partits_a_jugar.csv.
# Si el format del torneig canvia algun dia, només cal tocar aquest número.
FINAL_MATCH_ID = 104

MODEL_PATH  = "model_reg_log.joblib"
SCALER_PATH = "scaler_reg_log.joblib"
MATCHES_CSV = "partits_a_jugar.csv"
TEAMS_CSV   = "FIFA_2026_Data.csv"

# Quants millors tercers passen a la fase eliminatòria (format WC 2026: 48 equips, 8 tercers)
NUM_BEST_THIRDS = 8


# =============================================================================
# PREDICCIÓ D'UN PARTIT
# =============================================================================

def predict_match(
    team1: str,
    team2: str,
    team_data: dict,
    model,
    scaler,
) -> dict:
    """Retorna les probabilitats {classe: prob} per a un enfrontament team1 vs team2.

    Construïm les features com a DIFERÈNCIA entre equips perquè el model
    va ser entrenat exactament d'aquesta manera. Si passem els valors absoluts
    directament, el model retornaria probabilitats sense sentit.

    L'escalat és crític: sense scaler.transform les magnituds serien completament
    diferents de les del train i les probabilitats sortirien esbiaixades.
    """
    t1 = team_data[team1]
    t2 = team_data[team2]

    features = {
        "points_diff":        t1["points"]               - t2["points"],
        "points_sum":         t1["points"]               + t2["points"],
        "ppm_diff":           t1["points_per_match"]     - t2["points_per_match"],
        "gdpm_diff":          t1["dif_goals_per_match"]  - t2["dif_goals_per_match"],
        # same_confederation es fixa a 0 perquè en un mundial tots els grups
        # barrejen confederacions; si en algun cas coincidissin, caldria calcular-ho.
        "same_confederation": 0,
        # home_advantage ve de les dades de l'equip (0 si el terreny és neutral)
        "home_advantage":     t1.get("home_advantage", 0),
    }

    X = pd.DataFrame([features])[FEATURE_ORDER]
    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[0]
    return dict(zip(model.classes_, probs))


# =============================================================================
# SIMULACIÓ D'UN TORNEIG COMPLET
# =============================================================================

def simulate_tournament(df: pd.DataFrame, model, scaler, team_data: dict) -> str:
    """Simula un torneig sencer i retorna el nom del campió.

    El flux és:
      1. Fase de grups: cada equip acumula punts (3/1/0).
      2. Classificació: top-2 de cada grup + 8 millors tercers.
      3. Fase eliminatòria: cap empat possible; la probabilitat de l'empat
         es reparteix a parts iguals entre els dos equips.
    """

    # Diccionaris que anem omplint a mesura que avança el torneig
    team_points: dict[str, int] = {}
    match_winners: dict[int, str] = {}
    match_losers:  dict[int, str] = {}

    # --- FASE DE GRUPS ---
    group_stage = df[df["stage"] == "Group Stage"]

    for _, row in group_stage.iterrows():
        t1, t2 = row["team1"], row["team2"]

        # Inicialitzem a 0 la primera vegada que veiem un equip
        team_points.setdefault(t1, 0)
        team_points.setdefault(t2, 0)

        pred = predict_match(t1, t2, team_data, model, scaler)

        # L'ordre de les probabilitats ha de coincidir amb el de np.random.choice
        outcome = np.random.choice(
            [0, 1, 2],
            p=[pred.get("Draw", 0), pred.get("Team1", 0), pred.get("Team2", 0)],
        )

        if outcome == 1:       # Guanya Team1
            team_points[t1] += 3
        elif outcome == 2:     # Guanya Team2
            team_points[t2] += 3
        else:                  # Empat
            team_points[t1] += 1
            team_points[t2] += 1

    # --- CLASSIFICACIÓ DE FASE DE GRUPS ---
    standings: dict[str, str] = {}
    third_place_teams: list[tuple[str, int]] = []

    groups = df[df["stage"] == "Group Stage"]["group"].unique()

    for group in groups:
        # Recollim tots els equips del grup (apareixen a team1 i a team2)
        group_rows = df[df["group"] == group]
        teams_in_group = pd.concat([group_rows["team1"], group_rows["team2"]]).unique()

        sorted_teams = sorted(teams_in_group, key=lambda t: team_points.get(t, 0), reverse=True)

        # Guardem tant "winners" com "runners-up" perquè el CSV pot fer servir
        # qualsevol de les dues formes per referenciar el segon classificat.
        standings[f"{group} winners"]    = sorted_teams[0]
        standings[f"{group} runners-up"] = sorted_teams[1]
        standings[f"{group} runners up"] = sorted_teams[1]

        third_place_teams.append((sorted_teams[2], team_points.get(sorted_teams[2], 0)))

    # Agafem els NUM_BEST_THIRDS millors tercers i els barregem aleatòriament
    # per assignar-los als llocs de l'eliminatòria de manera imparcial.
    third_place_teams.sort(key=lambda x: x[1], reverse=True)
    best_thirds = [team for team, _ in third_place_teams[:NUM_BEST_THIRDS]]
    random.shuffle(best_thirds)

    # --- FASE ELIMINATÒRIA ---
    def resolve_team(name: str) -> str:
        """Tradueix una referència del CSV (ex: 'Winner match 50') a un nom d'equip real."""
        name = str(name).strip()
        if "winners" in name or "runners" in name:
            return standings.get(name, name)
        elif "third place" in name:
            # pop() agafa el darrer element; com que best_thirds ja està barrejat
            # és equivalent a un pop aleatori sense repetició.
            return best_thirds.pop() if best_thirds else "Unknown"
        elif "Winner match" in name:
            match_id = int(name.replace("Winner match", "").strip())
            return match_winners.get(match_id, name)
        elif "Runner-up match" in name:
            match_id = int(name.replace("Runner-up match", "").strip())
            return match_losers.get(match_id, name)
        return name

    knockouts = df[df["stage"] != "Group Stage"]

    for _, row in knockouts.iterrows():
        match_id = row["match_id"]
        t1 = resolve_team(row["team1"])
        t2 = resolve_team(row["team2"])

        pred = predict_match(t1, t2, team_data, model, scaler)

        prob_t1   = pred.get("Team1", 0)
        prob_draw = pred.get("Draw",  0)
        prob_t2   = pred.get("Team2", 0)

        # En eliminatòria no pot haver empat: repartim la seva probabilitat
        # a parts iguals entre els dos equips (equivalent a una pròrroga coin-flip).
        prob_t1 += prob_draw / 2.0
        prob_t2 += prob_draw / 2.0

        # Normalitzem per si els floats no sumen exactament 1.0 per precisió
        total = prob_t1 + prob_t2
        prob_t1 /= total
        prob_t2 /= total

        winner = np.random.choice([t1, t2], p=[prob_t1, prob_t2])
        loser  = t2 if winner == t1 else t1

        match_winners[match_id] = winner
        match_losers[match_id]  = loser

    return match_winners[FINAL_MATCH_ID]


# =============================================================================
# BUCLE MONTECARLO
# =============================================================================

def run_montecarlo(
    df_matches: pd.DataFrame,
    model,
    scaler,
    team_data: dict,
    num_simulations: int = 1000,
) -> Counter:
    """Executa num_simulations torneigs i retorna el comptador de victòries per equip.

    Separem la impressió dels resultats de la lògica de simulació perquè
    sigui fàcil reutilitzar aquesta funció en un context sense prints
    (ex: tests automàtics, API, notebook interactiu).
    """
    print(f"Iniciant {num_simulations} simulacions...")
    winners = []

    for i in range(num_simulations):
        champion = simulate_tournament(df_matches, model, scaler, team_data)
        winners.append(champion)

        # Imprimim progrés cada 100 simulacions per no saturar la consola
        if (i + 1) % 100 == 0:
            print(f"  -> Completades {i + 1} / {num_simulations} simulacions...")

    return Counter(winners)


def print_results(results: Counter, num_simulations: int, top_n: int = 15) -> None:
    """Imprimeix els resultats finals de forma llegible.

    Separem la presentació de la lògica per poder canviar el format
    (ex: exportar a CSV, mostrar en una UI) sense tocar res més.
    """
    print("\n" + "=" * 50)
    print("RESULTATS MONTECARLO — WC 2026")
    print("=" * 50)
    print(f"{'Equip':<25} | {'Prob. Guanyar':>13}")
    print("-" * 50)

    for team, wins in results.most_common(top_n):
        probability = (wins / num_simulations) * 100
        print(f"{team:<25} | {probability:>12.2f}%")


# =============================================================================
# PUNT D'ENTRADA
# =============================================================================

if __name__ == "__main__":
    scaler = joblib.load(SCALER_PATH)
    model  = joblib.load(MODEL_PATH)
    model.n_jobs = 1
    df_matches = pd.read_csv(MATCHES_CSV)
    # Convertim el CSV d'equips a un diccionari {nom_equip: {feature: valor}}
    team_data = pd.read_csv(TEAMS_CSV).set_index("team").to_dict(orient="index")
    results = run_montecarlo(df_matches, model, scaler, team_data, num_simulations=1000)
    print_results(results, num_simulations=1000)
