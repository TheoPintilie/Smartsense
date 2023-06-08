import numpy as np
import librosa
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("smartsense_model.h5")

# The bird species vector
bird_species = ["ana", "chira", "cormo",  "dum", "king", "pelican", "pes", "prigorie", "sil"]

# Define the parameters for the spectrogram
n_fft = 2048
hop_length = 512
n_mels = 128

# Load and preprocess the new audio file
file_path = "chira_013.wav"
y, sr = librosa.load(file_path)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_dB = librosa.power_to_db(S, ref=np.max)
S_dB = S_dB.reshape((n_mels, 130, 1))

# 130 - data shape

# Define the threshold for the minimum confidence prediction
threshold = 0.7

# Call the predict method of the model to obtain a probability distribution over the possible classes
predictions = model.predict(np.array([S_dB]))

# Create an array to store the predicted probabilities for all the classes
predicted_probabilities = predictions[0]

# Print the predicted class and its corresponding probability
if np.max(predicted_probabilities) >= threshold:
    print(
        f"The predicted class is: {bird_species[np.argmax(predicted_probabilities)]} with the confidence {np.max(predicted_probabilities) * 100}%")
else:
    print(f"The model is not confident enough to make a prediction with a minimum confidence of {threshold:.2f}.")

# Print the predicted class and probability for each class

print("Probability of all species:")

for i, species in enumerate(bird_species):
    print(f"     Probability of {species}: {predicted_probabilities[i]}")


