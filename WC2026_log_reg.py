# =============================================================================
# WC2026 — Regressió Logística per Predicció de Resultats
# =============================================================================
# Aquest script prepara les dades històriques de partits internacionals,
# integra el rànquing FIFA i estadístiques de forma, i entrena un model de regressió logística per predir el resultat (Team1 guanya, Team2 guanya, o empat). El model es guarda per a ús futur en la simulació del Mundial 2026.

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import cross_val_score


# =============================================================================
# 1. CÀRREGA I PREPARACIÓ DEL RÀNQUING FIFA
# =============================================================================
# Llegim el rànquing FIFA i ens quedem només amb les columnes que necessitem.
# Eliminem rank_change, previous_points, rank i country_abrv perquè no aporten
# informació nova: el que ens interessa és el pes FIFA (total_points), la
# confederació i la data per fer el merge_asof temporal més endavant.

df_ranking = (
    pd.read_csv("fifa_ranking-2024-06-20.csv")
    .drop(columns=["rank_change", "previous_points", "rank", "country_abrv"])
    .assign(rank_date=lambda x: pd.to_datetime(x["rank_date"]))
    .sort_values("rank_date")
)


# =============================================================================
# 2. CÀRREGA I NETEJA DELS PARTITS DE MUNDIALS
# =============================================================================
# Llegim tots els partits internacionals i filtrem només els mundials.
# La resta (amistosos, classificatòries, etc.) els gestionem a la secció 3.

# Diccionari de correccions: el fitxer de partits i el de rànquings no fan servir
# exactament els mateixos noms de país. Si no corregim això, el merge_asof
# no trobarà l'equip i generarà NaN als punts FIFA.
NOM_CORRECTIONS = {
    "Brunei": "Brunei Darussalam",
    "Cape Verde": "Cabo Verde",
    "China": "China PR",
    "DR Congo": "Congo DR",
    "East Timor": "Timor-Leste",
    "Eastern Samoa": "American Samoa",
    "Gambia": "The Gambia",
    "Iran": "IR Iran",
    "Ireland": "Republic of Ireland",
    "Ivory Coast": "Côte d'Ivoire",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Macao": "Macau",
    "Macedonia": "North Macedonia",
    "North Korea": "Korea DPR",
    "Saint Kitts": "St. Kitts and Nevis",
    "Saint Kitts and Nevis": "St. Kitts and Nevis",
    "Saint Lucia": "St. Lucia",
    "South Korea": "Korea Republic",
    "St Vincent & Grenadines": "St. Vincent and the Grenadines",
    "Swaziland": "Eswatini",
    "São Tome and Principe": "São Tomé and Príncipe",
    "Taiwan": "Chinese Taipei",
    "United States": "USA",
    "Western Samoa": "Samoa",
    "Curaçao": "Curacao",
}

raw_matches = pd.read_csv("all_matches.csv")

df_matches = (
    raw_matches[raw_matches["tournament"] == "World Cup"]
    .drop(columns=["country"])
    .assign(
        home_team=lambda x: x["home_team"].replace(NOM_CORRECTIONS),
        away_team=lambda x: x["away_team"].replace(NOM_CORRECTIONS),
        date=lambda x: pd.to_datetime(x["date"]),
    )
    # Filtrem des del 1993 perquè el rànquing FIFA va néixer el 1993;
    # partits anteriors no tindrien punts de referència vàlids.
    .loc[lambda x: x["date"] >= "1993-01-01"]
    .sort_values("date")
)


# =============================================================================
# 3. ESTADÍSTIQUES DE FORMA PER CICLE MUNDIALISTA (PPM i Diferència de Gols)
# =============================================================================
# Per no dependre únicament del rànquing FIFA estàtic, afegim mètriques
# de rendiment dins del cicle: punts per partit (ppm) i diferència de gols
# per partit (gdpm). Calculem-les sobre partits oficials (excloent mundials
# i amistosos) perquè reflecteixin la qualitat real de la classificatòria.

# --- 3a. Assignació del cicle mundialista ---
# Cada any pertany al cicle del proper mundial. Com que els mundials cauen
# en anys on (any % 4) == 2 (1994, 1998, … 2026), calculem el cicle amb
# aritmètica modular.
def assign_world_cup_cycle(year: int) -> int:
    """Retorna l'any del pròxim mundial donat un any qualsevol.

    Exemples: 2019 -> 2022, 2023 -> 2026, 2022 -> 2022.
    Els mundials cauen quan (any % 4) == 2, d'aquí els offsets.
    """
    offsets = {3: 3, 0: 2, 1: 1, 2: 0}
    return year + offsets[year % 4]


CYCLES_OF_INTEREST = {1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026}

df_qualifying = (
    raw_matches[
        (raw_matches["tournament"] != "Friendly")
        & (raw_matches["tournament"] != "World Cup")
    ]
    .drop(columns=["country"])
    .assign(
        home_team=lambda x: x["home_team"].replace(NOM_CORRECTIONS),
        away_team=lambda x: x["away_team"].replace(NOM_CORRECTIONS),
        date=lambda x: pd.to_datetime(x["date"], errors="coerce"),
    )
    .dropna(subset=["date"])
    .assign(year=lambda x: x["date"].dt.year.astype(int))
    .assign(wc_cycle=lambda x: x["year"].apply(assign_world_cup_cycle))
    .loc[lambda x: x["wc_cycle"].isin(CYCLES_OF_INTEREST)]
)

# --- 3b. Càlcul de punts i diferència de gols per equip i cicle ---
# Creem dues vistes (local/visitant), calculem els punts individuals
# i les concatenem per poder agrupar per equip.

def compute_team_stats(df: pd.DataFrame, team_col: str, score_col: str, opp_col: str) -> pd.DataFrame:
    """Calcula punts i diferència de gols per a un equip (local o visitant).

    Separem la lògica local/visitant en una funció per no repetir el mateix
    bloc de codi dues vegades (DRY). El cridarem un cop per cada perspectiva.
    """
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
        goal_diff_per_match=lambda x: (x["total_goal_diff"] / x["matches_played"]).round(2),
    )
    .reset_index()
    .sort_values(["wc_cycle", "points_per_match"], ascending=[False, False])
)


# =============================================================================
# 4. CONSTRUCCIÓ DEL DATASET PRINCIPAL
# =============================================================================
# Combinem partits + punts FIFA (temporalment correctes gràcies a merge_asof)
# + estadístiques del cicle de cada equip.

# --- 4a. Merge_asof: punts FIFA vigents en el moment del partit ---
# merge_asof és com un JOIN però busca el valor de rànquing més recent ANTERIOR
# a la data del partit (direction='backward'). Això evita data leakage: el model
# no "sap" els punts FIFA posteriors al partit que ha d'aprendre a predir.

ranking_lookup = df_ranking[["rank_date", "country_full", "confederation", "total_points"]]

df_with_rankings = (
    pd.merge_asof(
        df_matches,
        ranking_lookup,
        left_on="date", right_on="rank_date",
        left_by="home_team", right_by="country_full",
        direction="backward",
    )
    .rename(columns={"total_points": "points_team1", "confederation": "conf_team1"})
    .drop(columns=["rank_date", "country_full"])
)

df_with_rankings = (
    pd.merge_asof(
        df_with_rankings,
        ranking_lookup,
        left_on="date", right_on="rank_date",
        left_by="away_team", right_by="country_full",
        direction="backward",
    )
    .rename(columns={"total_points": "points_team2", "confederation": "conf_team2"})
    .drop(columns=["rank_date", "country_full"])
)

# Etiquetem el resultat de forma vectoritzada (molt més ràpid que apply/lambda per fila)
conditions = [
    df_with_rankings["home_score"] > df_with_rankings["away_score"],
    df_with_rankings["home_score"] < df_with_rankings["away_score"],
]
df_with_rankings["result"] = np.select(conditions, ["Team1", "Team2"], default="Draw")

# Eliminem les files sense punts FIFA per evitar entrenar amb dades incompletes
df_with_rankings = (
    df_with_rankings
    .dropna(subset=["points_team1", "points_team2"])
    .reset_index(drop=True)
)

# --- 4b. Afegir estadístiques del cicle ---
# Calculem el cicle mundialista del partit per fer el join correctament.
df_with_rankings["year"] = df_with_rankings["date"].dt.year
df_with_rankings["wc_cycle"] = df_with_rankings["year"].apply(assign_world_cup_cycle)

cycle_lookup = df_cycle_stats[["team", "wc_cycle", "points_per_match", "goal_diff_per_match"]]

df_full = (
    df_with_rankings
    .merge(cycle_lookup, left_on=["home_team", "wc_cycle"], right_on=["team", "wc_cycle"], how="left")
    .rename(columns={"points_per_match": "home_ppm", "goal_diff_per_match": "home_gdpm"})
    .drop(columns=["team"])
    .merge(cycle_lookup, left_on=["away_team", "wc_cycle"], right_on=["team", "wc_cycle"], how="left")
    .rename(columns={"points_per_match": "away_ppm", "goal_diff_per_match": "away_gdpm"})
    .drop(columns=["team"])
    .assign(neutral=lambda x: x["neutral"].astype(int))
)


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================
# Treballem amb diferències relatives en lloc de valors absoluts per dos motius:
# (a) les diferències capturen la força relativa entre equips, que és el que
#     de veritat determina qui guanya;
# (b) reduïm la colinearitat (points_team1 i points_team2 junts serien
#     redundants respecte a points_diff + points_sum).

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Afegeix les features derivades al DataFrame.

    Encapsulem el feature engineering en una funció perquè haurem de
    cridar-la exactament igual sobre train i test. Aixi evitem que
    qualsevol canvi es faci en un lloc i no en l'altre.
    """
    return df.assign(
        # Diferència i suma de punts FIFA: captura força relativa i absoluta
        points_diff=lambda x: x["points_team1"] - x["points_team2"],
        points_sum=lambda x: x["points_team1"] + x["points_team2"],
        # Forma del cicle: rendiment recent en classificatòria
        ppm_diff=lambda x: x["home_ppm"] - x["away_ppm"],
        gdpm_diff=lambda x: x["home_gdpm"] - x["away_gdpm"],
        # Partits de la mateixa confederació solen ser més igualats perquè
        # els equips es coneixen de les classificatòries
        same_confederation=lambda x: (x["conf_team1"] == x["conf_team2"]).astype(int),
        # Camp neutral elimina l'avantatge de local (neutral=1 -> home_advantage=0)
        home_advantage=lambda x: 1 - x["neutral"],
    )


# =============================================================================
# 6. DIVISIÓ TRAIN / TEST I DATA AUGMENTATION
# =============================================================================
# Tallem temporalment: entrenem fins a 2021 i testem amb 2022 en endavant.

CUTOFF_YEAR = 2022

# Eliminem features derivades que puguin existir a df_full per evitar
# duplicats quan cridem build_features() després del mirror.
DERIVED_COLS = ["points_diff", "points_sum", "ppm_diff", "gdpm_diff",
                "same_confederation", "home_advantage"]
df_base = df_full.drop(columns=[c for c in DERIVED_COLS if c in df_full.columns])

df_train_raw = df_base[df_base["year"] < CUTOFF_YEAR].copy()
df_test_raw  = df_base[df_base["year"] >= CUTOFF_YEAR].copy()


# --- 6a. Data Augmentation per simetria ---
# Sense mirror, el model mai veu un partit des de la perspectiva de l'equip
# visitant guanyant: la distribució seria Team1 >> Draw > Team2, forçant el
# model a ignorar sistemàticament els resultats de Team2.
# Amb el mirror, cada partit apareix dues vegades (A vs B i B vs A) i la
# distribució queda equilibrada (~50% T1, ~50% T2, Draw fix) sense necessitat
# de class_weight ni oversampling artificial.

MIRROR_PAIRS = [
    ("home_score",   "away_score"),
    ("points_team1", "points_team2"),
    ("conf_team1",   "conf_team2"),
    ("home_ppm",     "away_ppm"),
    ("home_gdpm",    "away_gdpm"),
]
RESULT_FLIP = {"Team1": "Team2", "Team2": "Team1", "Draw": "Draw"}


def apply_mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Duplica el DataFrame intercanviant equip local i visitant.

    El mirror s'aplica tant a train com a test. Si no ho féssim al test,
    els partits on Team2 és favorit mai apareixerien com a 'Team2 guanya'
    i el recall de Team2 cauria a zero a l'avaluació.
    """
    df_mirrored = df.copy()
    for col_a, col_b in MIRROR_PAIRS:
        if col_a in df_mirrored.columns and col_b in df_mirrored.columns:
            df_mirrored[[col_a, col_b]] = df_mirrored[[col_b, col_a]].values
    df_mirrored["result"] = df_mirrored["result"].map(RESULT_FLIP)
    return pd.concat([df, df_mirrored], ignore_index=True)


df_train_aug = (
    apply_mirror(df_train_raw)
    .sample(frac=1, random_state=42)  # Barregem per evitar patrons d'ordre artificial
    .reset_index(drop=True)
)
df_test_aug = apply_mirror(df_test_raw)  # NO barregem el test per mantenir reproductibilitat

print("Distribució classes — TRAIN augmentat:")
print(df_train_aug["result"].value_counts(normalize=True).round(3))
print("\nDistribució classes — TEST  augmentat:")
print(df_test_aug["result"].value_counts(normalize=True).round(3))


# --- 6b. Feature engineering sobre els conjunts augmentats ---
df_train_final = build_features(df_train_aug)
df_test_final  = build_features(df_test_aug)


# =============================================================================
# 7. PREPARACIÓ DE X / y PER AL MODEL
# =============================================================================
# Entren al model NOMÉS les features derivades. Les variables originals
# (punts absoluts, confederació, etc.) ja estan "codificades" a les derivades:
# afegir-les crearia multicolinearitat i podria confondre el model.

TARGET = "result"
COLS_TO_DROP = [
    TARGET, "home_score", "away_score", "date", "year", "wc_cycle",
    "home_team", "away_team", "tournament",
    "points_team1", "points_team2",
    "home_ppm", "away_ppm", "home_gdpm", "away_gdpm",
    "conf_team1", "conf_team2",
    "neutral",
]

y_train = df_train_final[TARGET]
y_test  = df_test_final[TARGET]

X_train = df_train_final.drop(columns=[c for c in COLS_TO_DROP if c in df_train_final.columns])
X_test  = df_test_final.drop(columns=[c for c in COLS_TO_DROP if c in df_test_final.columns])

print(f"\nX_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"Features: {X_train.columns.tolist()}")

# Escalem per evitar que features amb rangs molt grans (ex: points_diff pot
# arribar a ±800) dominin el gradient de la regressió logística.
# IMPORTANT: fit_transform NOMÉS sobre train; transform sobre test.
# Si féssim fit sobre test estaríem "filtrant" informació futura al model.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# =============================================================================
# 8. SELECCIÓ DE L'HIPERPARÀMETRE C (CROSS-VALIDATION)
# =============================================================================
# C controla la regularització L2: C petit = regularització forta (menys overfitting).
# Fem una cerca manual senzilla amb 5-fold CV per triar el millor valor.

print("\n--- Cross-Validation per seleccionar C ---")
for c in [0.5, 1.0, 2.0, 5.0, 10.0]:
    cv_model = LogisticRegression(solver="lbfgs", max_iter=2000, C=c, random_state=42)
    cv_score = cross_val_score(cv_model, X_train_scaled, y_train, cv=5, scoring="neg_log_loss").mean()
    print(f"  C={c:5.1f}  ->  log-loss CV: {-cv_score:.4f}")


# =============================================================================
# 9. ENTRENAMENT DEL MODEL FINAL
# =============================================================================
# Fem servir C=10.0 (poca regularització) perquè les features ja estan
# normalitzades i les diferències relatives tendeixen a tenir distribucions
# ben condicionades. Si la CV mostrés signes d'overfitting, baixaríem C.

model = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    C=10.0,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

predictions   = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)


# =============================================================================
# 10. AVALUACIÓ
# =============================================================================
accuracy = accuracy_score(y_test, predictions)
ll       = log_loss(y_test, probabilities)

print(f"\n{'='*52}")
print(f"  Accuracy : {accuracy * 100:.2f}%")
print(f"  Log-Loss : {ll:.4f}  (baseline naif ~1.099)")
print(f"{'='*52}\n")
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=model.classes_))

print("\nProbabilitats predites per a la primera mostra del test:")
for cls, prob in zip(model.classes_, probabilities[0]):
    print(f"  {cls:8s}: {prob:.3f}")


# =============================================================================
# 11. EXPORTACIÓ DEL MODEL I L'ESCALADOR
# =============================================================================

joblib.dump(model,  "model_reg_log.joblib")
joblib.dump(scaler, "scaler_reg_log.joblib")

print("\n✅ Model guardat a: model_reg_log.joblib")
print("✅ Escalador guardat a: scaler_reg_log.joblib")
