from flask import Flask, request, render_template, jsonify, current_app
import pandas as pd
import numpy as np
import utils # Your utils.py file
import logging

app = Flask(__name__)

# Configure logging
app.logger.setLevel(logging.INFO) 
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# Load artifacts when the Flask app starts
try:
    with app.app_context(): 
        current_app.logger.info("Attempting to load artifacts at startup...")
    utils.load_artifacts() 
    with app.app_context():
        current_app.logger.info("Artifacts loading process initiated from app.py.")
except Exception as e:
    with app.app_context():
        current_app.logger.error(f"FATAL: Could not load artifacts at startup: {e}", exc_info=True)

@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    if request.method == 'POST':
        try:
            # Get data from form using user-facing names
            raw_form_data = {
                # Demographics
                'Gender': request.form.get('gender'),
                'Age': request.form.get('age'),
                'Married': request.form.get('married'),
                'Number of Dependents': request.form.get('number_of_dependents'),
                'Number of Referrals': request.form.get('number_of_referrals'),
                # Account Information
                'Tenure in Months': request.form.get('tenure_in_months'),
                'Offer': request.form.get('offer'),
                'Contract': request.form.get('contract'),
                # Services
                'Phone Service': request.form.get('phone_service'),
                'Multiple Lines': request.form.get('multiple_lines'),
                'Internet Service': request.form.get('internet_service'),
                'Internet Type': request.form.get('internet_type'),
                'Online Security': request.form.get('online_security'),
                'Online Backup': request.form.get('online_backup'),
                'Device Protection Plan': request.form.get('device_protection_plan'),
                'Premium Tech Support': request.form.get('premium_tech_support'),
                'Streaming TV': request.form.get('streaming_tv'),
                'Streaming Movies': request.form.get('streaming_movies'),
                'Streaming Music': request.form.get('streaming_music'),
                'Unlimited Data': request.form.get('unlimited_data'),
                # Charges and Billing
                # User enters "Average Monthly Roaming Charges"
                'Average Monthly Roaming Charges': request.form.get('avg_monthly_roaming_charges'),
                'Avg Monthly GB Download': request.form.get('avg_monthly_gb_download'),
                'Monthly Charge': request.form.get('monthly_charge'), 
                'Total Refunds': request.form.get('total_refunds'),     
                'Total Extra Data Charges': request.form.get('total_extra_data_charges'),
                # "Total Roaming Charges" is no longer a direct form input
                'Paperless Billing': request.form.get('paperless_billing'),
                'Payment Method': request.form.get('payment_method'),
            }
            current_app.logger.info(f"Raw form data received: {raw_form_data}")

            # Convert to snake_case and prepare data for utils.py
            processed_input_for_utils = {}
            avg_monthly_roaming_charges_str = raw_form_data.get('Average Monthly Roaming Charges', '0') # Default to '0' if missing

            for key, value in raw_form_data.items():
                snake_key = key.lower().strip().replace(' ', '_')
                if snake_key == 'average_monthly_roaming_charges':
                    # This input maps to the original 'avg_monthly_long_distance_charges' feature
                    processed_input_for_utils['avg_monthly_long_distance_charges'] = value 
                else:
                    processed_input_for_utils[snake_key] = value
            
            # Calculate 'total_long_distance_charges' (internal name for total roaming)
            # based on the average monthly roaming charges.
            try:
                avg_roaming_float = float(avg_monthly_roaming_charges_str if avg_monthly_roaming_charges_str else 0.0)
            except ValueError:
                avg_roaming_float = 0.0
                current_app.logger.warning(f"Could not convert avg_monthly_roaming_charges '{avg_monthly_roaming_charges_str}' to float. Defaulting to 0.0.")

            calculated_total_roaming = avg_roaming_float * 3.2
            processed_input_for_utils['total_long_distance_charges'] = calculated_total_roaming # This is the internal name for the total
            
            # Add calculated total roaming to raw_form_data for display on results page if desired
            raw_form_data['Calculated Total Roaming Charges'] = f"{calculated_total_roaming:.2f}"


            current_app.logger.info(f"Data prepared for utils.py: {processed_input_for_utils}")

            processed_df = utils.preprocess_input(processed_input_for_utils)
            current_app.logger.info("Input preprocessed successfully by utils.py.")

            prediction_label, churn_probability = utils.predict_churn(processed_df)
            current_app.logger.info(f"Prediction made: Label='{prediction_label}', Probability='{churn_probability}'")

            churn_probability_percent = f"{churn_probability*100:.2f}%"
            
            return render_template('results.html',
                                   prediction=prediction_label,
                                   probability_raw=churn_probability,
                                   probability_percent=churn_probability_percent,
                                   raw_input=raw_form_data)

        except FileNotFoundError as e:
            current_app.logger.error(f"Artifact file not found: {e}", exc_info=True)
            return render_template('error.html', error_message=f"A required model file was not found: {e}.")
        except ValueError as e: 
            current_app.logger.error(f"Value error during processing: {e}", exc_info=True)
            return render_template('error.html', error_message=f"Invalid input data: {e}")
        except Exception as e:
            current_app.logger.error(f"An error occurred during prediction: {e}", exc_info=True)
            import traceback
            tb_str = traceback.format_exc()
            return render_template('error.html', error_message=f"An unexpected error occurred: {e}\n<pre>{tb_str}</pre>")

    return "Invalid request method.", 405

# ... (all your existing Flask app code above this line remains unchanged) ...

if __name__ == '__main__':
    from waitress import serve
    # Make sure your Flask app instance is named 'app'
    # For example, you should have a line like: app = Flask(__name__) somewhere above
    host = '127.0.0.1'
    port = 5000 # You can change this port if needed

    # This line will be printed in the console if you run the .exe with --console
    # or if you run python app.py directly
    print(f"Starting server with Waitress on http://{host}:{port}") 

    serve(app, host=host, port=port)