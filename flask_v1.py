import os
import pyaudio
import wave
import librosa
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    output_dir = 'recordings'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'output.wav')

    frames = []
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 22050
    DURATION = 3.00  # Desired duration in seconds
    CHUNKS_PER_SECOND = int(RATE / CHUNK)  # Number of chunks per second
    NUM_CHUNKS = int(DURATION * CHUNKS_PER_SECOND)  # Total number of chunks for desired duration

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    for _ in range(NUM_CHUNKS):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Adjust the duration of the recording
    frames[-1] += b'\x00' * (len(frames[-1]) - len(data))

    with wave.open(output_file, 'wb') as wf:
        # Write the recorded audio frames to the file
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return 'Recording saved'

    audio_data = b''.join(frames)
    # Process the recorded audio using librosa or save it as a WAV file
    process_audio(audio_data)

    return 'Audio recorded and processed'

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    if file and allowed_file(file.filename):
        audio_data = file.read()
        # Process the uploaded audio using librosa or save it as a WAV file
        process_audio(audio_data)

        return 'File uploaded and processed successfully'

    return 'Invalid file format', 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'wav'

def process_audio(audio_data):
    # Use librosa to process the audio data
    # Replace the following code with your desired audio processing logic
    audio, sr = librosa.load(audio_data, sr=None)
    # Perform further processing or analysis on the loaded audio

if __name__ == '__main__':
    app.run(debug=True)