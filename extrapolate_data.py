import pandas as pd
import numpy as np
import os
import time

def extrapolate_dataset(input_path: str, output_path: str, target_size: int = 50000):
    print(f"Starting Data Extrapolation Pipeline...")
    start_time = time.time()
    
    # 1. Load original data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    original_size = len(df)
    
    # 2. Drop duplicates to find truly unique patients (ignoring PatientID)
    cols_to_check = [c for c in df.columns if c != 'PatientID']
    df_unique = df.drop_duplicates(subset=cols_to_check)
    unique_size = len(df_unique)
    
    print(f"Original size: {original_size} rows")
    print(f"Unique rows found: {unique_size} rows")
    print(f"Dropped {original_size - unique_size} identical duplicates.")
    
    if unique_size >= target_size:
        print("Unique rows already exceed target size. No need to extrapolate.")
        return

    # Calculate how many new rows we need to generate
    num_to_generate = target_size - unique_size
    print(f"Generating {num_to_generate} brand new synthetic patient records...")
    
    # Separate features by type
    categorical_cols = df_unique.select_dtypes(include=['object', 'category']).columns.tolist()
    # also consider features with very few unique values as categorical
    for col in df_unique.columns:
        if col not in categorical_cols and df_unique[col].nunique() <= 5:
            categorical_cols.append(col)
            
    continuous_cols = [c for c in df_unique.columns if c not in categorical_cols and c not in ['PatientID']]
    
    # 3. Generate synthetic data
    # We will randomly sample from the unique rows with replacement, and then add realistic "jitter" (noise)
    synthetic_samples = df_unique.sample(n=num_to_generate, replace=True).copy()
    
    # Add statistical jitter to continuous columns (e.g. +- 2% of standard deviation)
    # This makes the numbers unique but clinically identical in meaning
    for col in continuous_cols:
        std_dev = df_unique[col].std()
        if pd.isna(std_dev) or std_dev == 0:
            continue
            
        # Add random normal noise: mean=0, std = 2% of the feature's standard deviation
        noise = np.random.normal(0, std_dev * 0.02, size=num_to_generate)
        synthetic_samples[col] = synthetic_samples[col] + noise
        
        # Ensure we don't create impossible values (like negative age or blood pressure)
        min_val = df_unique[col].min()
        max_val = df_unique[col].max()
        synthetic_samples[col] = np.clip(synthetic_samples[col], min_val, max_val)
        
        # If the original data was integers (like Age=50), keep the synthetic data as integers
        if pd.api.types.is_integer_dtype(df_unique[col]):
            synthetic_samples[col] = synthetic_samples[col].round().astype(int)

    # For categorical columns, we'll introduce a tiny 1% mutation rate to prevent exact feature matches
    for col in categorical_cols:
        # Get the distribution of categories
        value_counts = df_unique[col].value_counts(normalize=True)
        categories = value_counts.index.tolist()
        probabilities = value_counts.values.tolist()
        
        if len(categories) > 1:
            # 1% chance to mutate the category to a random other category based on real distribution
            mutate_mask = np.random.random(size=num_to_generate) < 0.01
            num_mutations = mutate_mask.sum()
            
            if num_mutations > 0:
                random_cats = np.random.choice(categories, size=num_mutations, p=probabilities)
                synthetic_samples.loc[mutate_mask, col] = random_cats

    # Assign new Patient IDs
    max_existing_id = 0
    if 'PatientID' in df_unique.columns:
        # Assuming PatientID is numeric or can be stripped to numeric
        try:
            max_existing_id = int(df_unique['PatientID'].max())
        except:
            max_existing_id = 100000
            
        synthetic_samples['PatientID'] = range(max_existing_id + 1, max_existing_id + 1 + num_to_generate)

    # 4. Combine real unique data with synthetic data
    df_final = pd.concat([df_unique, synthetic_samples], ignore_index=True)
    
    # Shuffle the dataset
    df_final = df_final.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # 5. Save the new extrapolated dataset
    print(f"Saving new dataset with {len(df_final)} truly unique/synthetic rows to {output_path}...")
    df_final.to_csv(output_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"Extrapolation complete in {elapsed:.2f} seconds!")
    print("You can now update data_processor.py to use this new file if you wish.")

if __name__ == "__main__":
    input_file = "Chronickidneydiseases.csv"
    output_file = "Chronickidneydiseases_synthetic_50k.csv"
    
    if os.path.exists(input_file):
        extrapolate_dataset(input_file, output_file, target_size=50000)
    else:
        print(f"Error: Could not find {input_file} in the current directory.")
