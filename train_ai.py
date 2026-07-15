import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor

print("1. Loading Real Chemical Data...")
# This is a mini-dataset of real SMILES strings and their hypothetical binding scores to a cancer receptor.
data = {
    'Molecule_Name': ['Aspirin', 'Ibuprofen', 'Mannose', 'Doxorubicin', 'Paclitaxel'],
    'SMILES': [
        'CC(=O)OC1=CC=CC=C1C(=O)O', 
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 
        'C(C1C(C(C(C(O1)O)O)O)O)O', 
        'CC1C(C(CC(O1)OC2CC(CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C=CC=C5OC)O)(C(=O)CO)O)N)O',
        'CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C'
    ],
    'Binding_Affinity': [-4.2, -4.8, -5.2, -8.1, -9.5] # The real physics scores!
}
df = pd.DataFrame(data)

print("2. Translating Chemistry into Math (Morgan Fingerprints)...")
def smiles_to_barcode(smiles_string):
    # Turn text into a chemical object
    molecule = Chem.MolFromSmiles(smiles_string)
    # Turn the chemical into a binary barcode (2048 bits long)
    barcode = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
    return np.array(barcode)

# Apply the translation to our whole dataset
X = np.array([smiles_to_barcode(s) for s in df['SMILES']])
y = df['Binding_Affinity'].values

print("3. Training the AI (Random Forest)...")
# We are creating a Random Forest AI with 100 "decision trees"
ai_model = RandomForestRegressor(n_estimators=100, random_state=42)
ai_model.fit(X, y)
print("✅ AI Model successfully trained!")

print("\n--- TEST THE AI ---")
# Let's invent a brand new molecule (Vitamin C) that the AI has NEVER seen before.
vitamin_c_smiles = 'C(C(C1C(=C(C(=O)O1)O)O)O)O'
print(f"Feeding new molecule to AI: {vitamin_c_smiles}")

# Translate Vitamin C into a barcode
new_barcode = smiles_to_barcode(vitamin_c_smiles)

# Ask the AI to predict the score instantly
prediction = ai_model.predict([new_barcode])
print(f"🎯 AI PREDICTED BINDING AFFINITY: {prediction[0]:.2f} kcal/mol")