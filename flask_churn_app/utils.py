import pandas as pd
import numpy as np
import joblib
import json

# --- Global cache for artifacts ---
ARTIFACTS = {}

def load_artifacts():
    """Loads all necessary artifacts into a global dictionary."""
    global ARTIFACTS
    if ARTIFACTS: # Avoid reloading if already loaded
        return

    print("Loading artifacts (v5 - 'number_of_referrals' feature removed)...")
    try:
        ARTIFACTS['model'] = joblib.load('model.pkl')
        ARTIFACTS['scaler'] = joblib.load('scaler.pkl')
        ARTIFACTS['imputer'] = joblib.load('imputer.pkl')
        ARTIFACTS['label_encoder'] = joblib.load('label_encoder.pkl')

        with open('label_encoder_classes.json', 'r') as f:
            ARTIFACTS['label_encoder_classes'] = json.load(f)
        with open('x_columns_before_imputation.json', 'r') as f:
            ARTIFACTS['x_columns_before_imputation'] = json.load(f)
        with open('outlier_bounds.json', 'r') as f:
            ARTIFACTS['outlier_bounds'] = json.load(f)
        with open('numerical_columns_for_scaling.json', 'r') as f:
            ARTIFACTS['numerical_columns_for_scaling'] = json.load(f)
        with open('final_model_columns.json', 'r') as f:
            ARTIFACTS['final_model_columns'] = json.load(f)
        with open('skew_transform_map.json', 'r') as f:
            ARTIFACTS['skew_transform_map'] = json.load(f)
        print("All artifacts loaded successfully.")
        print("Reminder: This version expects artifacts generated AFTER 'number_of_referrals' was dropped.")
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}. Make sure all .pkl and .json files are in the root directory and are updated.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during artifact loading: {e}")
        raise

def preprocess_input(raw_input_dict):
    """
    Preprocesses a dictionary of raw input features into a format
    suitable for the churn prediction model.
    'number_of_referrals' is no longer expected.
    """
    if not ARTIFACTS:
        load_artifacts()

    print("Preprocessing input (v5)...")
    df = pd.DataFrame([raw_input_dict]) # raw_input_dict keys are already snake_case

    print(f"Received columns for preprocessing: {df.columns.tolist()}")

    # 2. Data type conversion
    # 'number_of_referrals' is removed from this list
    numeric_form_inputs = {
        'age': int, 'number_of_dependents': int, # 'number_of_referrals': int, REMOVED
        'tenure_in_months': int, 'avg_monthly_long_distance_charges': float,
        'avg_monthly_gb_download': float, 'monthly_charge': float,
        'total_refunds': float, 'total_extra_data_charges': float,
        'total_long_distance_charges': float # This is the internal name for total roaming
    }

    for col, dtype in numeric_form_inputs.items():
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if pd.api.types.is_integer_dtype(pd.Series(0, dtype=dtype)):
                    df[col] = df[col].fillna(0).astype(dtype)
                else:
                    df[col] = df[col].astype(dtype)
            except ValueError as e:
                print(f"Warning: Could not convert column '{col}' to {dtype}. Error: {e}. Setting to 0 or NaN.")
                df[col] = 0 if pd.api.types.is_integer_dtype(pd.Series(0, dtype=dtype)) else np.nan
        # else: # This case should not happen if app.py sends all expected keys (excluding referrals)
            # print(f"Warning: Expected numeric column '{col}' not found in input dictionary.")


    # --- Calculate 'total_revenue' ---
    monthly_charge_val = df.get('monthly_charge', pd.Series(0.0, index=df.index)).fillna(0).astype(float)
    tenure_in_months_val = df.get('tenure_in_months', pd.Series(0, index=df.index)).fillna(0).astype(int)
    total_extra_data_charges_val = df.get('total_extra_data_charges', pd.Series(0.0, index=df.index)).fillna(0).astype(float)
    total_long_distance_charges_val = df.get('total_long_distance_charges', pd.Series(0.0, index=df.index)).fillna(0).astype(float)
    total_refunds_val = df.get('total_refunds', pd.Series(0.0, index=df.index)).fillna(0).astype(float)

    df['total_revenue'] = (monthly_charge_val * tenure_in_months_val) + \
                           total_extra_data_charges_val + \
                           total_long_distance_charges_val - \
                           total_refunds_val
    print(f"Calculated total_revenue: {df['total_revenue'].iloc[0] if not df.empty else 'N/A'}")


    # 3. Skewness Transformations
    # skew_transform_map.json should have been updated by save_artifacts_script_v2.py
    # to exclude 'number_of_referrals'.
    skew_map = ARTIFACTS['skew_transform_map']
    for col_original, transform_type in skew_map.items():
        if col_original in df.columns: # This will skip 'number_of_referrals' if not in map
            df[col_original] = pd.to_numeric(df[col_original], errors='coerce').fillna(0)
            if transform_type == 'sqrt':
                df[f'{col_original}_sqrt'] = np.sqrt(np.maximum(0, df[col_original]))
            elif transform_type == 'log':
                df[f'{col_original}_log'] = np.log1p(df[col_original])
    print(f"Columns after skew transformations (if any): {df.columns.tolist()}")


    # 4. Null Value Filling & Logic for Dependent Services (remains the same logic)
    is_internet_service_no = df.get('internet_service', pd.Series(dtype=str)).iloc[0] == 'No'

    if 'internet_type' in df.columns:
        if is_internet_service_no:
            df['internet_type'] = 'no_internet_service'
        else:
            df['internet_type'] = df['internet_type'].fillna('no_internet_service')
            if df['internet_type'].iloc[0] == '': 
                 df['internet_type'] = 'no_internet_service'

    if 'offer' in df.columns:
        df['offer'] = df['offer'].fillna('no_offer')

    home_internet_features = ['online_security', 'online_backup', 'device_protection_plan',
                              'premium_tech_support', 'streaming_tv', 'streaming_movies',
                              'streaming_music', 'unlimited_data']
    for col in home_internet_features:
        if col in df.columns:
            if is_internet_service_no:
                df[col] = 'no_internet_service'
            else:
                df[col] = df[col].fillna('no_internet_service')
                if df[col].iloc[0] == '': 
                    df[col] = 'no_internet_service'

    if 'avg_monthly_gb_download' in df.columns:
        if is_internet_service_no:
            df['avg_monthly_gb_download'] = 0.0
        else:
            df['avg_monthly_gb_download'] = pd.to_numeric(df['avg_monthly_gb_download'], errors='coerce').fillna(0.0)

    if 'avg_monthly_long_distance_charges' in df.columns:
         df['avg_monthly_long_distance_charges'] = pd.to_numeric(df['avg_monthly_long_distance_charges'], errors='coerce').fillna(0.0)
    print("Null values and dependent service logic applied.")


    # 5. 'multiple_lines' mapping (remains the same logic)
    if 'multiple_lines' in df.columns:
        is_phone_service_no = df.get('phone_service', pd.Series(dtype=str)).iloc[0] == 'No'
        if is_phone_service_no:
            df['multiple_lines'] = 0.0
        else:
            def map_multiple_lines(val):
                if isinstance(val, str):
                    val_lower = val.lower()
                    if val_lower == 'yes': return 1.0
                    if val_lower == 'no': return 0.0
                    if val_lower == 'no phone service': return 0.0
                    try: return float(val)
                    except ValueError: return np.nan
                return float(val) if pd.notna(val) else np.nan
            df['multiple_lines'] = df['multiple_lines'].apply(map_multiple_lines)
    print("'multiple_lines' mapped.")

    # 6. Get Dummies for categorical features
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    categorical_cols_for_dummies = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols_for_dummies:
        df = pd.get_dummies(df, columns=categorical_cols_for_dummies, drop_first=True)
    print(f"Columns after get_dummies: {df.columns.tolist()}")


    # 7. Align columns with `x_columns_before_imputation.json`
    # This JSON file should have been updated by save_artifacts_script_v2.py
    expected_cols_for_imputer = ARTIFACTS['x_columns_before_imputation']
    for col in expected_cols_for_imputer:
        if col not in df.columns:
            df[col] = 0 
            print(f"Added missing column for imputer: {col} (value 0)")
    cols_to_drop = [col for col in df.columns if col not in expected_cols_for_imputer]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped extra columns not expected by imputer: {cols_to_drop}")
    df = df[expected_cols_for_imputer] 
    print(f"Columns aligned for KNNImputer: {len(df.columns)}")


    # 8. KNN Imputation
    imputer = ARTIFACTS['imputer'] # Imputer should have been refit by save_artifacts_script_v2.py
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    imputed_values = imputer.transform(df)
    df = pd.DataFrame(data=imputed_values, columns=df.columns)
    print("KNN Imputation applied.")


    # 9. Round 'multiple_lines' after imputation
    if 'multiple_lines' in df.columns:
        df['multiple_lines'] = df['multiple_lines'].round().astype(int)
    print("'multiple_lines' rounded after imputation.")


    # 10. Outlier Handling
    # outlier_bounds.json should have been updated by save_artifacts_script_v2.py
    outlier_bounds = ARTIFACTS['outlier_bounds']
    for col, bounds in outlier_bounds.items(): 
        if col in df.columns:
            df[col] = np.where(df[col] < bounds['lower'], bounds['lower'], df[col])
            df[col] = np.where(df[col] > bounds['upper'], bounds['upper'], df[col])
    print("Outliers handled.")


    # 11. Scaling numerical features
    # numerical_columns_for_scaling.json should have been updated.
    # scaler.pkl should have been refit.
    scaler = ARTIFACTS['scaler']
    numerical_cols_to_scale = ARTIFACTS['numerical_columns_for_scaling']
    actual_numerical_cols_in_df = [col for col in numerical_cols_to_scale if col in df.columns]
    if actual_numerical_cols_in_df:
        df[actual_numerical_cols_in_df] = df[actual_numerical_cols_in_df].fillna(0)
        df[actual_numerical_cols_in_df] = scaler.transform(df[actual_numerical_cols_in_df])
        print(f"Numerical features scaled: {actual_numerical_cols_in_df}")
    else:
        print("Warning: No numerical columns (from list) found in DataFrame to scale.")


    # 12. Final column alignment to match model's training columns
    # final_model_columns.json should have been updated.
    final_model_cols = ARTIFACTS['final_model_columns']
    for col in final_model_cols: 
        if col not in df.columns:
            df[col] = 0 
            print(f"Added missing column for final model: {col} (value 0)")
    cols_to_drop_final = [col for col in df.columns if col not in final_model_cols] 
    if cols_to_drop_final:
        df = df.drop(columns=cols_to_drop_final)
        print(f"Dropped extra columns for final model: {cols_to_drop_final}")
    df = df[final_model_cols] 
    print(f"Final columns for model ({len(df.columns)}): {df.columns.tolist()}")
    
    print("Preprocessing complete.")
    return df


def predict_churn(processed_data_df):
    """Makes a churn prediction using the loaded model."""
    if not ARTIFACTS:
        load_artifacts()

    model = ARTIFACTS['model'] # Model.pkl should be the newly trained one
    label_encoder_classes = ARTIFACTS['label_encoder_classes']

    try:
        prediction_numeric = model.predict(processed_data_df)
        prediction_proba = model.predict_proba(processed_data_df)
        predicted_label = label_encoder_classes[prediction_numeric[0]]
        prob_churn = 0.0
        try:
            churn_label_from_encoder = 'Churned'
            if churn_label_from_encoder in label_encoder_classes:
                churn_class_index = label_encoder_classes.index(churn_label_from_encoder)
                prob_churn = prediction_proba[0][churn_class_index]
            else: 
                print(f"Warning: Positive class '{churn_label_from_encoder}' not found by string in label_encoder_classes {label_encoder_classes}. Attempting to infer index.")
                if len(label_encoder_classes) > 1 and prediction_proba.shape[1] > 1:
                    prob_churn = prediction_proba[0][1] 
                    print(f"Assuming 'Churned' corresponds to index 1 of probabilities. Proba: {prediction_proba[0]}")
                elif prediction_proba.shape[1] > 0 : 
                    prob_churn = prediction_proba[0][0]
                else:
                    prob_churn = 0.0 
        except Exception as e_proba:
            print(f"Warning: Error determining churn probability for '{churn_label_from_encoder}': {e_proba}. Defaulting to probability of predicted class.")
            prob_churn = prediction_proba[0][prediction_numeric[0]]
        return predicted_label, prob_churn
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

if __name__ == '__main__':
    print("Testing utils.py (v5 - 'number_of_referrals' removed)...")
    try:
        load_artifacts()
        print(f"Artifacts loaded for test: {list(ARTIFACTS.keys())}")
        sample_raw_input_internal = {
            'gender': 'Male', 'age': 30, 'married': 'Yes', 'number_of_dependents': 0,
            # 'number_of_referrals': 0, # REMOVED
            'tenure_in_months': 1, 'offer': 'None',
            'phone_service': 'Yes', 
            'avg_monthly_long_distance_charges': 10.50, 
            'multiple_lines': 'No', 
            'internet_service': 'Yes', 
            'internet_type': 'DSL',
            'avg_monthly_gb_download': 10, 
            'online_security': 'No', 'online_backup': 'No',
            'device_protection_plan': 'No', 'premium_tech_support': 'No', 'streaming_tv': 'No',
            'streaming_movies': 'No', 'streaming_music': 'No', 'unlimited_data': 'No',
            'contract': 'Month-to-Month', 'paperless_billing': 'Yes',
            'payment_method': 'Bank Withdrawal', 'monthly_charge': 20.0,
            'total_refunds': 0.0, 'total_extra_data_charges': 0.0, 
            'total_long_distance_charges': (10.50 * 3.2) # Example: avg_roaming * 3.2 for internal total
        }
        print(f"\nSample Raw Input (Internal Names, no referrals): {sample_raw_input_internal}")

        processed_df = preprocess_input(sample_raw_input_internal.copy())
        print(f"\nProcessed DataFrame shape: {processed_df.shape}")
        
        if not processed_df.empty:
            label, probability = predict_churn(processed_df)
            print(f"\nPrediction: Label={label}, Churn Probability={probability:.4f}")
        else:
            print("\nProcessed DataFrame is empty. Prediction skipped.")
    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
        import traceback
        traceback.print_exc()
