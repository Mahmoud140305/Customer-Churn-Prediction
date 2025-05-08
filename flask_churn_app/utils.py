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

    print("Loading artifacts...")
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
        print("Reminder: This version uses 'total_long_distance_charges' internally based on existing artifacts.")
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}. Make sure all .pkl and .json files are in the root directory.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during artifact loading: {e}")
        raise

def preprocess_input(raw_input_dict):
    """
    Preprocesses a dictionary of raw input features into a format
    suitable for the churn prediction model.
    Assumes 'total_long_distance_charges' is the key used internally,
    mapped from 'total_roaming_charges' in app.py if needed.
    """
    if not ARTIFACTS:
        load_artifacts()

    print("Preprocessing input...")
    # raw_input_dict already has 'total_long_distance_charges' if mapped in app.py
    df = pd.DataFrame([raw_input_dict])


    # 1. Column names are already snake_case from app.py (or should be)
    # df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns] # This might be redundant if app.py handles it
    print(f"Received columns for preprocessing: {df.columns.tolist()}")


    # 2. Data type conversion
    # 'total_long_distance_charges' is the internal name expected by artifacts
    numeric_form_inputs = {
        'age': int, 'number_of_dependents': int, 'number_of_referrals': int,
        'tenure_in_months': int, 'avg_monthly_long_distance_charges': float,
        'avg_monthly_gb_download': float, 'monthly_charge': float,
        'total_refunds': float, 'total_extra_data_charges': float,
        'total_long_distance_charges': float # This is the internal name
    }

    for col, dtype in numeric_form_inputs.items():
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Convert to numeric, coerce errors to NaN
                if pd.api.types.is_integer_dtype(pd.Series(0, dtype=dtype)): # Check if target is int
                    df[col] = df[col].fillna(0).astype(dtype) # Fill NaN with 0 for int types
                else: # Float types
                    df[col] = df[col].astype(dtype) # Let NaNs pass for float, imputer will handle
            except ValueError as e:
                print(f"Warning: Could not convert column '{col}' to {dtype}. Error: {e}. Setting to 0 or NaN.")
                df[col] = 0 if pd.api.types.is_integer_dtype(pd.Series(0, dtype=dtype)) else np.nan


    # --- Calculate 'total_revenue' ---
    # Monthly charge includes all services except roaming charges (internally 'total_long_distance_charges').
    # Total Refunds are for the period.
    # Formula: (monthly_charge * tenure_in_months) + total_extra_data_charges + total_long_distance_charges - total_refunds

    monthly_charge_val = df.get('monthly_charge', pd.Series(0.0, index=df.index)).fillna(0).astype(float)
    tenure_in_months_val = df.get('tenure_in_months', pd.Series(0, index=df.index)).fillna(0).astype(int)
    total_extra_data_charges_val = df.get('total_extra_data_charges', pd.Series(0.0, index=df.index)).fillna(0).astype(float)
    # Use the internal name 'total_long_distance_charges'
    total_long_distance_charges_val = df.get('total_long_distance_charges', pd.Series(0.0, index=df.index)).fillna(0).astype(float)
    total_refunds_val = df.get('total_refunds', pd.Series(0.0, index=df.index)).fillna(0).astype(float)

    df['total_revenue'] = (monthly_charge_val * tenure_in_months_val) + \
                           total_extra_data_charges_val + \
                           total_long_distance_charges_val - \
                           total_refunds_val
    print(f"Calculated total_revenue: {df['total_revenue'].iloc[0] if not df.empty else 'N/A'}")


    # 3. Skewness Transformations
    # Artifacts (skew_transform_map.json) expect original column names like 'total_long_distance_charges'.
    skew_map = ARTIFACTS['skew_transform_map']
    for col_original, transform_type in skew_map.items():
        if col_original in df.columns:
            df[col_original] = pd.to_numeric(df[col_original], errors='coerce').fillna(0)
            if transform_type == 'sqrt':
                df[f'{col_original}_sqrt'] = np.sqrt(np.maximum(0, df[col_original]))
            elif transform_type == 'log':
                df[f'{col_original}_log'] = np.log1p(df[col_original])
    print(f"Columns after skew transformations (if any): {df.columns.tolist()}")


    # 4. Null Value Filling & Logic for Dependent Services
    is_internet_service_no = df.get('internet_service', pd.Series(dtype=str)).iloc[0] == 'No'

    if 'internet_type' in df.columns:
        if is_internet_service_no:
            df['internet_type'] = 'no_internet_service'
        else:
            # If form sends empty for internet_type when Internet Service is Yes, treat as 'no_internet_service'
            # or a specific "Unknown" if your model was trained with it.
            # Your original script used .apply(lambda x: 'no_internet_service' if pd.isnull(x) else x)
            df['internet_type'] = df['internet_type'].fillna('no_internet_service')
            if df['internet_type'].iloc[0] == '': # Handle empty string from form
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
                if df[col].iloc[0] == '': # Handle empty string from form
                    df[col] = 'no_internet_service'


    if 'avg_monthly_gb_download' in df.columns:
        if is_internet_service_no:
            df['avg_monthly_gb_download'] = 0.0
        else:
            # Ensure it's numeric, fillna if it became NaN during type conversion
            df['avg_monthly_gb_download'] = pd.to_numeric(df['avg_monthly_gb_download'], errors='coerce').fillna(0.0)

    if 'avg_monthly_long_distance_charges' in df.columns: # This is distinct from total_long_distance_charges (roaming)
         df['avg_monthly_long_distance_charges'] = pd.to_numeric(df['avg_monthly_long_distance_charges'], errors='coerce').fillna(0.0)

    print("Null values and dependent service logic applied.")


    # 5. 'multiple_lines' mapping
    if 'multiple_lines' in df.columns:
        is_phone_service_no = df.get('phone_service', pd.Series(dtype=str)).iloc[0] == 'No'
        if is_phone_service_no:
            # If 'No phone service' was a distinct category that became a dummy variable,
            # this needs to align. For now, mapping to 0 (like 'No').
            # Or, if 'multiple_lines_No phone service' was a dummy, that should be 1.
            # Your original code: data['multiple_lines'] = data['multiple_lines'].map({"Yes": 1, "No": 0})
            # It did not seem to handle "No phone service" as a category for multiple_lines directly before KNN.
            # So, if phone service is No, multiple_lines is effectively No.
            df['multiple_lines'] = 0.0
        else:
            def map_multiple_lines(val):
                if isinstance(val, str):
                    val_lower = val.lower()
                    if val_lower == 'yes': return 1.0
                    if val_lower == 'no': return 0.0
                    if val_lower == 'no phone service': return 0.0 # Map this explicitly if it comes from form
                    try: return float(val) # For "1" or "0"
                    except ValueError: return np.nan # If unmappable string
                return float(val) if pd.notna(val) else np.nan # For numeric or NaN
            df['multiple_lines'] = df['multiple_lines'].apply(map_multiple_lines)
            # If after mapping it's still NaN (e.g. empty string from form for active phone service)
            # KNN imputer will handle it if 'multiple_lines' is float.
            # Your original code did KNN impute on this after mapping.
    print("'multiple_lines' mapped.")


    # 6. Get Dummies for categorical features
    for col in df.select_dtypes(include=['object']).columns: # Ensure all object columns are strings
        df[col] = df[col].astype(str)

    categorical_cols_for_dummies = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols_for_dummies:
        df = pd.get_dummies(df, columns=categorical_cols_for_dummies, drop_first=True)
    print(f"Columns after get_dummies: {df.columns.tolist()}")


    # 7. Align columns with `x_columns_before_imputation`
    expected_cols_for_imputer = ARTIFACTS['x_columns_before_imputation']
    
    # Add missing columns (expected by imputer but not in current df after dummies)
    for col in expected_cols_for_imputer:
        if col not in df.columns:
            df[col] = 0 # Or np.nan. Your KNNImputer was fit on data that had undergone get_dummies.
                        # If a category for a dummy wasn't present in this new instance, its column won't exist.
                        # Adding it as 0 is standard.
            print(f"Added missing column for imputer: {col} (value 0)")

    # Drop extra columns (in current df but not expected by imputer)
    # This can happen if a new, unexpected categorical value was passed that wasn't in training.
    # However, get_dummies with known columns from training (via reindex) is more robust.
    # Here, we rely on the `expected_cols_for_imputer` list.
    cols_to_drop = [col for col in df.columns if col not in expected_cols_for_imputer]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped extra columns not expected by imputer: {cols_to_drop}")
    
    df = df[expected_cols_for_imputer] # Ensure correct order and exact set of columns
    print(f"Columns aligned for KNNImputer: {len(df.columns)}")


    # 8. KNN Imputation
    imputer = ARTIFACTS['imputer']
    # Ensure all data is numeric for imputer; NaNs are fine.
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
    outlier_bounds = ARTIFACTS['outlier_bounds']
    for col, bounds in outlier_bounds.items(): # col here must match keys in outlier_bounds.json
        if col in df.columns:
            df[col] = np.where(df[col] < bounds['lower'], bounds['lower'], df[col])
            df[col] = np.where(df[col] > bounds['upper'], bounds['upper'], df[col])
    print("Outliers handled.")


    # 11. Scaling numerical features
    scaler = ARTIFACTS['scaler']
    numerical_cols_to_scale = ARTIFACTS['numerical_columns_for_scaling'] # Must match keys in .json
    
    actual_numerical_cols_in_df = [col for col in numerical_cols_to_scale if col in df.columns]
    if actual_numerical_cols_in_df:
        df[actual_numerical_cols_in_df] = df[actual_numerical_cols_in_df].fillna(0) # Should not be needed after KNN
        df[actual_numerical_cols_in_df] = scaler.transform(df[actual_numerical_cols_in_df])
        print(f"Numerical features scaled: {actual_numerical_cols_in_df}")
    else:
        print("Warning: No numerical columns (from list) found in DataFrame to scale.")


    # 12. Final column alignment to match model's training columns
    final_model_cols = ARTIFACTS['final_model_columns'] # Must match keys in .json
    
    for col in final_model_cols: # Add any missing columns expected by the model
        if col not in df.columns:
            df[col] = 0 
            print(f"Added missing column for final model: {col} (value 0)")
            
    cols_to_drop_final = [col for col in df.columns if col not in final_model_cols] # Drop any extra
    if cols_to_drop_final:
        df = df.drop(columns=cols_to_drop_final)
        print(f"Dropped extra columns for final model: {cols_to_drop_final}")
        
    df = df[final_model_cols] # Ensure exact order and set of columns for the model
    print(f"Final columns for model ({len(df.columns)}): {df.columns.tolist()}")
    
    print("Preprocessing complete.")
    return df


def predict_churn(processed_data_df):
    """Makes a churn prediction using the loaded model."""
    if not ARTIFACTS:
        load_artifacts()

    model = ARTIFACTS['model']
    label_encoder_classes = ARTIFACTS['label_encoder_classes'] # e.g., ['Stayed', 'Churned']

    try:
        prediction_numeric = model.predict(processed_data_df)
        prediction_proba = model.predict_proba(processed_data_df)

        predicted_label = label_encoder_classes[prediction_numeric[0]]
        
        prob_churn = 0.0
        try:
            # Find the index of the 'Churned' class.
            churn_label_from_encoder = 'Churned' # The positive class label string
            if churn_label_from_encoder in label_encoder_classes:
                churn_class_index = label_encoder_classes.index(churn_label_from_encoder)
                prob_churn = prediction_proba[0][churn_class_index]
            # Example: if le.classes_ is ['Stayed', 'Churned'], Churned is index 1.
            # If le.classes_ is ['Churned', 'Stayed'], Churned is index 0.
            # Your original code: data['customer_status'] = label_encoder.fit_transform(data['customer_status'])
            # data['customer_status'].value_counts() -> Stayed: 4720, Churned: 1869
            # If fit_transform sees ['Stayed', 'Churned', 'Stayed', ...], classes_ will be sorted: ['Churned', 'Stayed'] or ['Stayed', 'Churned']
            # Let's assume based on typical ordering, 'Churned' might be index 0 if it appears first alphabetically or by first encounter.
            # Or it could be index 1. It's critical to know what label_encoder.transform(['Churned']) yielded.
            # For now, relying on finding the string 'Churned'.
            else: # Fallback if 'Churned' string is not directly in classes_
                print(f"Warning: Positive class '{churn_label_from_encoder}' not found by string in label_encoder_classes {label_encoder_classes}. Attempting to infer index.")
                # This is a guess. If 'Churned' was encoded as 1, and 'Stayed' as 0, then prob_churn is proba[0][1]
                # If label_encoder_classes is like [0, 1] (numeric labels), this needs to map to the correct one.
                # Check your label_encoder.transform(['Churned']) in notebook.
                # A common convention: if 'Churned' is the positive class, it's often encoded as 1.
                if len(label_encoder_classes) > 1 and prediction_proba.shape[1] > 1:
                     # This assumes 'Churned' is the class at index 1 if not found by string. This is a common convention.
                    prob_churn = prediction_proba[0][1]
                    print(f"Assuming 'Churned' corresponds to index 1 of probabilities. Proba: {prediction_proba[0]}")
                elif prediction_proba.shape[1] > 0 : # Only one probability value
                    prob_churn = prediction_proba[0][0]
                else:
                    prob_churn = 0.0 # Should not happen

        except Exception as e_proba:
            print(f"Warning: Error determining churn probability for '{churn_label_from_encoder}': {e_proba}. Defaulting to probability of predicted class.")
            prob_churn = prediction_proba[0][prediction_numeric[0]]


        return predicted_label, prob_churn
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

if __name__ == '__main__':
    # This test block needs to be run from the root of your project
    # where the .pkl and .json files are located.
    print("Testing utils.py...")
    try:
        load_artifacts()
        print(f"Artifacts loaded for test: {list(ARTIFACTS.keys())}")

        # Sample input for testing. Keys should be as received from app.py
        # (i.e., snake_case, and 'total_long_distance_charges' is the internal name)
        sample_raw_input_internal = {
            'gender': 'Male', 'age': 30, 'married': 'Yes', 'number_of_dependents': 0,
            'number_of_referrals': 0, 'tenure_in_months': 1, 'offer': 'None',
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
            'total_long_distance_charges': 5.00 # Internal name
        }
        print(f"\nSample Raw Input (Internal Names): {sample_raw_input_internal}")

        processed_df = preprocess_input(sample_raw_input_internal.copy())
        print(f"\nProcessed DataFrame shape: {processed_df.shape}")
        # print(f"Processed DataFrame columns: {processed_df.columns.tolist()}")
        # print(f"Processed DataFrame head:\n{processed_df.head()}")

        if not processed_df.empty:
            label, probability = predict_churn(processed_df)
            print(f"\nPrediction: Label={label}, Churn Probability={probability:.4f}")
        else:
            print("\nProcessed DataFrame is empty. Prediction skipped.")

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
        import traceback
        traceback.print_exc()

