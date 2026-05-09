import json
import sys

# Load the notebook
with open('Train_Music_Genre_Classifier.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with the load_wav_file function
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and any('def load_wav_file' in line for line in cell['source']):
        # Replace the source
        cell['source'] = [
            'import os\n',
            'import librosa\n',
            'import matplotlib.pyplot as plt\n',
            'import tensorflow as tf\n',
            'import numpy as np\n',
            'import audioread\n',
            'import librosa.display\n',
            'from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout\n',
            'from tensorflow.keras.optimizers.legacy import Adam\n',
            '\n',
            'def load_wav_file(path):\n',
            '    # Use librosa.load with audioread backend (now available)\n',
            '    y, sr = librosa.load(path, sr=None)\n',
            '    # Ensure finite values\n',
            '    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)\n',
            '    return y, sr\n'
        ]
        break

# Save the notebook
with open('Train_Music_Genre_Classifier.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully")