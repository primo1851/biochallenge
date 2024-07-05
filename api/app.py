import os
import pickle
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import pandas as pd
import tensorflow as tf
from extractFeature import extract_features  # Assuming this function extracts features from audio

# Flask application setup
app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

# Endpoint for predicting output based on input data
class GetPredictionOutput(Resource):
    def post(self):
        try:
            # Example URL to download a test audio file (you can replace with your own)
            audio_url = 'https://storage.googleapis.com/audioset/yamalyzer/audio/speech.wav'
            testing_wav_file_name = tf.keras.utils.get_file('speech.wav',
                                                            audio_url,
                                                            cache_dir='./',
                                                            cache_subdir='test_data')

            # Extract features from the downloaded audio file
            new_audio_features = extract_features(testing_wav_file_name)
            new_audio_df = pd.DataFrame([new_audio_features])  # Convert to DataFrame if needed
            
            # Load the random forest model
            model_filepath = 'random_forest_model_av1.pkl'  # Update with your actual filepath
           
            

            
            with open(model_filepath, 'rb') as f:
                loaded_classifier = pickle.load(f)

           

            return jsonify({'predict': loaded_classifier})  # Convert predictions to list if needed

        except FileNotFoundError as fnf_error:
            return jsonify({'error': f"Model file not found: {str(fnf_error)}"})
        except Exception as e:
            return jsonify({'error': str(e)})

# Adding resources to the API
api.add_resource(Test, '/')  # Endpoint for testing GET request
api.add_resource(GetPredictionOutput, '/getPredictionOutput')  # Endpoint for predicting output based on input data

# Run the Flask application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
