import pandas as pd
import numpy as np

def clean_chembl_data(input_file, target_name, output_file):
    print(f"🧹 Loading raw data from {input_file}...")
    
    # 1. Load the messy ChEMBL CSV
    # (Using low_memory=False because ChEMBL files are huge)
    raw_df = pd.read_csv(input_file, sep=';', low_memory=False) 
    
    # 2. Keep only the columns we actually care about
    # Note: ChEMBL uses 'Smiles' and 'Standard Value'
    if 'Smiles' not in raw_df.columns or 'Standard Value' not in raw_df.columns:
        print("❌ Error: Could not find 'Smiles' or 'Standard Value' columns.")
        return
        
    clean_df = raw_df[['Smiles', 'Standard Value', 'Standard Type']].copy()
    
    # 3. Filter for IC50, Kd, or Ki values (the actual binding scores)
    valid_types = ['IC50', 'Kd', 'Ki']
    clean_df = clean_df[clean_df['Standard Type'].isin(valid_types)]
    
    # 4. Drop any rows where the SMILES or the Score is missing
    clean_df = clean_df.dropna(subset=['Smiles', 'Standard Value'])
    
    # 5. Rename columns to match our Streamlit App's AI
    clean_df = clean_df.rename(columns={
        'Smiles': 'SMILES',
        'Standard Value': 'Binding_Affinity'
    })
    
    # 6. Add our Target Name column
    clean_df['Target'] = target_name
    
    # 7. Save the beautiful, clean data
    clean_df[['SMILES', 'Target', 'Binding_Affinity']].to_csv(output_file, index=False)
    print(f"✅ Success! Cleaned data saved to {output_file}. Ready for Streamlit!")

# --- HOW TO USE THIS SCRIPT ---
# 1. Download the raw CSV from ChEMBL for EGFR. Let's say it's named "raw_egfr.csv".
# 2. Run this function:
# clean_chembl_data("raw_egfr.csv", "Lung (EGFR)", "clean_egfr_data.csv")