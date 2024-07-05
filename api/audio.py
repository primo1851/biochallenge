import pyaudio
import wave

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
return WAVE_OUTPUT_FILENAME