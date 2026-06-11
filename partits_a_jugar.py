import pandas as pd
import numpy as np

df = pd.read_csv('FIFA2026_schedule_Fixtures.csv')
df = df.drop(columns=['date', 'match_number', 'stadium', 'date_dt'])
# Separar la columna 'teams' en dues columnes: 'team1' i 'team2'
# Posem espais al voltant de la 'v' (' v ') perquè no ens quedi l'espai en el nom de l'equip
df[['team1', 'team2']] = df['teams'].str.split(' v ', expand=True)
# Eliminar la columna original 'teams' que ja no necessitem
df = df.drop(columns=['teams'])

equips_classificats = {
    "Czechia/Denmark/North Macedonia/Republic of Ireland": "Czechia",
    "Bosnia and Herzegovina/Italy/Northern Ireland/Wales": "Bosnia and Herzegovina",
    "Kosovo/Romania/Slovakia/Türkiye": "Türkiye",
    "Albania/Poland/Sweden/Ukraine": "Sweden",
    "Bolivia/Iraq/Suriname": "Iraq",
    "Congo DR/Jamaica/New Caledonia": "Congo DR"
}

# Substituïm aquestes cadenes llargues pels equips guanyadors a les columnes d'equips
df['team1'] = df['team1'].replace(equips_classificats)
df['team2'] = df['team2'].replace(equips_classificats)

# Afegim un ID de partit (el Mundial 2026 té 104 partits)
# Això és VITAL per a la simulació, per saber quin "Winner match X" estem referenciant.
df.insert(0, 'match_id', range(1, len(df) + 1))

# Afegim una columna per determinar la fase del torneig
def assignar_fase(match_id):
    if match_id <= 72: return "Group Stage"
    elif match_id <= 88: return "Round of 32"
    elif match_id <= 96: return "Round of 16"
    elif match_id <= 100: return "Quarter-finals"
    elif match_id <= 102: return "Semi-finals"
    elif match_id == 103: return "Third Place"
    else: return "Final"

df.insert(1, 'stage', df['match_id'].apply(assignar_fase))

# Neteja final: On la columna "group" és buida (NaN), hi posem "Knockout".
# I netegem els espais en blanc innecessaris dels noms d'equips.
df['group'] = df['group'].fillna("Knockout")
df['team1'] = df['team1'].str.strip()
df['team2'] = df['team2'].str.strip()

# Mostrem un resum del resultat
print("--- PRIMERS 5 PARTITS (Fase de Grups) ---")
print(df.head())
print("\n--- ÚLTIMS 5 PARTITS (Fase Final) ---")
print(df.tail())

df.to_csv('partits_a_jugar.csv', index=False)
print("Dades transformades i guardades a 'partits_a_jugar.csv'")