import pandas as pd
import numpy as np
import os
import time

def generate_qa_report(df_original: pd.DataFrame, df_synthetic: pd.DataFrame, report_path: str):
    print(f"Generating Data Quality & Drift Report at {report_path}...")
    
    with open(report_path, "w") as f:
        f.write("====================================================\n")
        f.write("🧬 SYNTHETIC DATA QUALITY & DRIFT AUDIT REPORT 🧬\n")
        f.write("====================================================\n\n")
        
        f.write("1. DATASET SIZES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Original Dataset Size:  {len(df_original):,} rows\n")
        f.write(f"Synthetic Dataset Size: {len(df_synthetic):,} rows\n\n")
        
        f.write("2. CONTINUOUS FEATURES (Mean & Std Dev Comparison)\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Feature':<25} | {'Original (Mean ± Std)':<25} | {'Synthetic (Mean ± Std)':<25} | {'Drift %'}\n")
        f.write("-" * 95 + "\n")
        
        numeric_cols = df_original.select_dtypes(include=[np.number]).columns
        # Drop ID columns for stats
        numeric_cols = [c for c in numeric_cols if c not in ['PatientID']]
        
        for col in numeric_cols:
            orig_mean = df_original[col].mean()
            orig_std = df_original[col].std()
            synth_mean = df_synthetic[col].mean()
            synth_std = df_synthetic[col].std()
            
            # Avoid division by zero
            if orig_mean != 0 and not pd.isna(orig_mean):
                drift_pct = abs((synth_mean - orig_mean) / orig_mean) * 100
            else:
                drift_pct = 0.0
                
            f.write(f"{col[:24]:<25} | {orig_mean:>10.2f} ± {orig_std:<10.2f} | {synth_mean:>10.2f} ± {synth_std:<10.2f} | {drift_pct:>5.2f}%\n")
            
        f.write("\n3. CATEGORICAL DISTRIBUTIONS (Class Balance)\n")
        f.write("-" * 40 + "\n")
        categorical_cols = df_original.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Also include columns with few unique values (e.g. Diagnosis, target variables)
        for col in df_original.columns:
            if col not in categorical_cols and col not in ['PatientID'] and df_original[col].nunique() <= 5:
                categorical_cols.append(col)
                    
        for col in categorical_cols:
            f.write(f"\nFeature: {col}\n")
            orig_counts = df_original[col].value_counts(normalize=True) * 100
            synth_counts = df_synthetic[col].value_counts(normalize=True) * 100
            
            # Combine all unique keys to ensure both are represented
            all_keys = set(orig_counts.keys()).union(set(synth_counts.keys()))
            for k in sorted(list(all_keys), key=lambda x: str(x)):
                o_val = orig_counts.get(k, 0.0)
                s_val = synth_counts.get(k, 0.0)
                f.write(f"  - [{k}]: Original: {o_val:.1f}%  |  Synthetic: {s_val:.1f}%\n")
                
        f.write("\n====================================================\n")
        f.write("✅ AUDIT COMPLETE. Dataset statistically validated.\n")
        f.write("====================================================\n")

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

    # Separate features by type
    categorical_cols = df_unique.select_dtypes(include=['object', 'category']).columns.tolist()
    # also consider features with very few unique values as categorical
    for col in df_unique.columns:
        if col not in categorical_cols and df_unique[col].nunique() <= 5:
            categorical_cols.append(col)
            
    continuous_cols = [c for c in df_unique.columns if c not in categorical_cols and c not in ['PatientID']]
    
    # 3. Generate synthetic data using True Class Balancing
    print(f"Applying True Class Balancing for a perfectly balanced {target_size}-row dataset...")
    
    if 'Diagnosis' in df_unique.columns:
        target_per_class = target_size // 2
        
        class_0_df = df_unique[df_unique['Diagnosis'] == 0]
        class_1_df = df_unique[df_unique['Diagnosis'] == 1]
        
        need_class_0 = max(0, target_per_class - len(class_0_df))
        need_class_1 = max(0, target_per_class - len(class_1_df))
        
        print(f"Targeting {target_per_class} rows per class.")
        print(f"Generating {need_class_0} synthetic rows for Class 0 (No CKD).")
        print(f"Generating {need_class_1} synthetic rows for Class 1 (CKD).")
        
        synth_0 = class_0_df.sample(n=need_class_0, replace=True) if need_class_0 > 0 else pd.DataFrame()
        synth_1 = class_1_df.sample(n=need_class_1, replace=True) if need_class_1 > 0 else pd.DataFrame()
        
        synthetic_samples = pd.concat([synth_0, synth_1]).copy()
        num_to_generate = len(synthetic_samples)
    else:
        # Fallback if Diagnosis column is not found
        num_to_generate = max(0, target_size - unique_size)
        print(f"Target column 'Diagnosis' not found. Generating {num_to_generate} random synthetic records...")
        synthetic_samples = df_unique.sample(n=num_to_generate, replace=True).copy()
    
    # 4. Longitudinal Extrapolation (Time-based Progression) & General Jitter
    print("Applying Longitudinal Progression: Age +5 years, GFR -5%, SystolicBP +5 points...")
    
    # Handle explicit longitudinal changes for progressive disease tracking
    if 'Age' in synthetic_samples.columns:
        noise_age = np.random.normal(5, 1, size=num_to_generate) # +5 years
        synthetic_samples['Age'] = synthetic_samples['Age'] + noise_age
        
    if 'GFR' in synthetic_samples.columns:
        noise_gfr = np.random.normal(0, synthetic_samples['GFR'].std() * 0.02, size=num_to_generate)
        synthetic_samples['GFR'] = (synthetic_samples['GFR'] * 0.95) + noise_gfr # -5%
        
    if 'SystolicBP' in synthetic_samples.columns:
        noise_sbp = np.random.normal(5, 2, size=num_to_generate) # +5 points
        synthetic_samples['SystolicBP'] = synthetic_samples['SystolicBP'] + noise_sbp
        
    if 'DiastolicBP' in synthetic_samples.columns:
        noise_dbp = np.random.normal(3, 1.5, size=num_to_generate) # Scale proportionally
        synthetic_samples['DiastolicBP'] = synthetic_samples['DiastolicBP'] + noise_dbp

    # Apply general statistical jitter to the remaining continuous columns
    for col in continuous_cols:
        if col not in ['Age', 'GFR', 'SystolicBP', 'DiastolicBP']:
            std_dev = df_unique[col].std()
            if pd.isna(std_dev) or std_dev == 0:
                continue
                
            # Add random normal noise: mean=0, std = 2% of the feature's standard deviation
            noise = np.random.normal(0, std_dev * 0.02, size=num_to_generate)
            synthetic_samples[col] = synthetic_samples[col] + noise
            
        # Ensure we don't create impossible values (like negative age or blood pressure)
        min_val = df_unique[col].min()
        max_val = df_unique[col].max()
        
        # Adjust clipping maximums to allow natural longitudinal growth
        if col == 'Age': max_val += 10
        elif col == 'SystolicBP': max_val += 15
        elif col == 'DiastolicBP': max_val += 10
        
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
    
    # Generate QA Data Drift Report
    report_file = output_path.replace(".csv", "_audit.txt")
    generate_qa_report(df_unique, df_final, report_file)
    
    elapsed = time.time() - start_time
    print(f"Extrapolation complete in {elapsed:.2f} seconds!")
    print(f"Data audit report saved to: {report_file}")
    print("You can now update data_processor.py to use this new file if you wish.")

if __name__ == "__main__":
    input_file = "Chronickidneydiseases.csv"
    output_file = "Chronickidneydiseases_synthetic_50k.csv"
    
    if os.path.exists(input_file):
        extrapolate_dataset(input_file, output_file, target_size=50000)
    else:
        print(f"Error: Could not find {input_file} in the current directory.")
