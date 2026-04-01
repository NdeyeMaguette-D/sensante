"""
Génération du dataset patients_dakar.csv
500 patients fictifs - SénSanté
"""
import pandas as pd
import random

random.seed(42)

regions = ["dakar", "thies", "saint-louis", "ziguinchor", 
           "tambacounda", "kaolack", "fatick", "louga", "diourbel", "matam"]

diagnostics = ["paludisme", "grippe", "typhoide", "sain"]

data = []

for i in range(500):
    diag = random.choices(diagnostics, weights=[27, 26, 15, 32])[0]
    
    if diag == "paludisme":
        temp = round(random.uniform(38.8, 40.5), 1)
        toux, fatigue, maux_tete = random.randint(0,1), 1, 1
        frissons, nausee = 1, random.randint(0,1)
    elif diag == "grippe":
        temp = round(random.uniform(38.0, 39.5), 1)
        toux, fatigue, maux_tete = 1, 1, random.randint(0,1)
        frissons, nausee = random.randint(0,1), random.randint(0,1)
    elif diag == "typhoide":
        temp = round(random.uniform(38.5, 40.0), 1)
        toux, fatigue, maux_tete = 0, 1, 1
        frissons, nausee = random.randint(0,1), 1
    else:  # sain
        temp = round(random.uniform(36.1, 37.5), 1)
        toux, fatigue, maux_tete = 0, 0, 0
        frissons, nausee = 0, 0

    data.append({
        "age": random.randint(5, 80),
        "sexe": random.choice(["M", "F"]),
        "temperature": temp,
        "tension_sys": random.randint(9, 16),
        "toux": toux,
        "fatigue": fatigue,
        "maux_tete": maux_tete,
        "frissons": frissons,
        "nausee": nausee,
        "region": random.choice(regions),
        "diagnostic": diag
    })

df = pd.DataFrame(data)
df.to_csv("data/patients_dakar.csv", index=False)
print(f"Dataset généré : {len(df)} patients")
print(df["diagnostic"].value_counts())
print("Fichier sauvegardé dans data/patients_dakar.csv ✅")