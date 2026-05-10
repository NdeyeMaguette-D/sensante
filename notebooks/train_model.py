import pandas as pd
import numpy as np

# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")
# Renommer les diagnostics pour correspondre au TP
df['diagnostic'] = df['diagnostic'].replace({
    'paludisme': 'palu',
    'typhoide': 'typh'
})

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

from sklearn.preprocessing import LabelEncoder

# Encoder les variables categoriques en nombre 
le_sexe   = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded']   = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Definir les features (X) et la cible (y)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]   
y = df['diagnostic']   

print(f"Features : {X.shape}")  # (500, 8)
print(f"Cible : {y.shape}")     # (500 ,)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% pour le test
    random_state=42,    # pour avoir les mêmes résultats à chaque fois
    stratify=y          # garder les mêmes proportions de diagnostics
)

print(f"Entraînement : {X_train.shape[0]} patients")  # → 400
print(f"Test         : {X_test.shape[0]} patients")   # → 100

from sklearn.ensemble import RandomForestClassifier

# Creer le modele
model = RandomForestClassifier(
    n_estimators=100,   # 100 arbres de décision
    random_state=42     # Reproductibilite
)

# Entrainer sur les donnees d'entrainement
model.fit(X_train, y_train)   

print("Modèle entraîné !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

# Predire sur les donnees de test
y_pred = model.predict(X_test)

# Comparer les 10 premieres predictions avec la realite 
comparison = pd.DataFrame({ 
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})
print(comparison)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy: .2%}")


from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Visualiser avec seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel('Prédiction du modèle')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SénSanté')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()

print("Figure sayvegardee dans figures/confusion_matrix.png")

import joblib
import os

# Creer le dossier models/ s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Sauvegarder le modèle
joblib.dump(model, "models/model.pkl")

# Verifier la taille du fichier 
size = os.path.getsize("models/model.pkl")
print(f"Modele sauvegarde : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

# Sauvegarder aussi les encodeurs — INDISPENSABLE !
joblib.dump(le_sexe,      "models/encoder_sexe.pkl")
joblib.dump(le_region,    "models/encoder_region.pkl")

# Sauvegarder la liste des features (pour references)
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardes.")


# Simuler ce que fera l'API en lab 3 :
# Charger le medele DEPUIS LE FICHIER (pas depuis la memoire)
model_loaded     = joblib.load("models/model.pkl")
le_sexe_loaded   = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# Un nouveau patient arrive au centre de sante de Medina
nouveau_patient = {
    'age': 28, 'sexe': 'F', 'temperature': 39.5,
    'tension_sys': 110, 'toux': True,
    'fatigue': True, 'maux_tete': True, 'region': 'Dakar'
}

# Encoder les valeurs categoriques
sexe_enc   = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Construire le vecteur de features
features = [
    nouveau_patient['age'], sexe_enc,
    nouveau_patient['temperature'], nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']), int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']), region_enc
]

# Prédire
diagnostic = model_loaded.predict([features])[0]
probas     = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print(f"Diagnostic  : {diagnostic}")
print(f"Probabilité : {probas.max():.1%}")

for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:8s} : {proba:.1%} {bar}")

# Exercice 1 : Importance des features
print("\n--- Importance des features ---")
importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances),
                        key=lambda x: x[1], reverse=True):
    print(f"  {name:20s} : {imp:.3f}")

# Exercice 2 : 3 patients fictifs
print("\n--- Exercice 2 : 3 patients fictifs ---")

patients_fictifs = [
    {'age': 20, 'sexe': 'M', 'temperature': 36.8, 'tension_sys': 120,
     'toux': False, 'fatigue': False, 'maux_tete': False, 'region': 'Dakar',
     'description': 'Jeune sans symptomes'},
    {'age': 45, 'sexe': 'F', 'temperature': 40.2, 'tension_sys': 105,
     'toux': True, 'fatigue': True, 'maux_tete': True, 'region': 'Dakar',
     'description': 'Adulte avec forte fievre'},
    {'age': 68, 'sexe': 'M', 'temperature': 37.9, 'tension_sys': 118,
     'toux': True, 'fatigue': True, 'maux_tete': False, 'region': 'Dakar',
     'description': 'Patient age avec toux'},
]

for p in patients_fictifs:
    s_enc = le_sexe_loaded.transform([p['sexe']])[0]
    r_enc = le_region_loaded.transform([p['region']])[0]
    feat  = [p['age'], s_enc, p['temperature'], p['tension_sys'],
             int(p['toux']), int(p['fatigue']), int(p['maux_tete']), r_enc]
    diag  = model_loaded.predict([feat])[0]
    prob  = model_loaded.predict_proba([feat])[0].max()
    print(f"  [{p['description']}] -> {diag} ({prob:.0%})") 