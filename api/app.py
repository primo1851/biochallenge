import os
import pickle
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import pandas as pd
import tensorflow as tf
from extractFeature import extract_features  # Assuming this function extracts features from audio
import json
import pyaudio
import wave

# Flask application setup
app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)


# Configurações para a gravação de áudio
FORMAT = pyaudio.paInt16        # Formato de codificação de áudio (16 bits PCM)
CHANNELS = 1                    # Número de canais de áudio (1 para mono, 2 para estéreo)
RATE = 44100                    # Taxa de amostragem (número de amostras por segundo)
CHUNK = 1024                    # Número de quadros por bloco de leitura/gravação
RECORD_SECONDS = 10              # Duração da gravação em segundos
WAVE_OUTPUT_FILENAME = "output.wav"  # Nome do arquivo de saída .wav

# Inicializar o objeto PyAudio
audio = pyaudio.PyAudio()

# Abrir o stream de gravação
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Gravando áudio...")

frames = []

# Gravar áudio em chunks e salvar frames
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Gravação concluída.")

# Parar e fechar o stream
stream.stop_stream()
stream.close()
audio.terminate()

# Salvar os frames capturados como um arquivo WAV
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Áudio salvo como {WAVE_OUTPUT_FILENAME}")

# Endpoint for predicting output based on input data
class GetPredictionOutput(Resource):
    def post(self):
        try:
            # Example URL to download a test audio file (you can replace with your own)
            audio_url = WAVE_OUTPUT_FILENAME
            testing_wav_file_name = tf.keras.utils.get_file(WAVE_OUTPUT_FILENAME,
                                                            audio_url,
                                                            cache_dir='./',
                                                            cache_subdir='test_data')

            # Extract features from the downloaded audio file

            print (testing_wav_file_name)
            new_audio_features = extract_features(testing_wav_file_name)
            new_audio_df = pd.DataFrame([new_audio_features])  # Convert to DataFrame if needed
            
            # Load the random forest model
            model_filepath = 'random_forest_model_av1.pkl'  # Update with your actual filepath
            
            with open(model_filepath, 'rb') as f:
                loaded_classifier = pickle.load(f)

            result = json.dumps(loaded_classifier.tolist())

            return jsonify({'predict': result})  # Convert predictions to list if needed

        except FileNotFoundError as fnf_error:
            return jsonify({'error': f"Model file not found: {str(fnf_error)}"})
        except Exception as e:
            return jsonify({'error': str(e)})

# Adding resources to the API
api.add_resource(GetPredictionOutput, '/getPredictionOutput')  # Endpoint for predicting output based on input data

# Run the Flask application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
