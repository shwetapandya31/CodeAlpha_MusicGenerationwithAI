# AI Music Generator Project by Shweta Pandya

This project is an AI-powered music generation system built using PyTorch and the `music21` library. Uniquely designed, it learns musical patterns from a dataset of custom MIDI files, trains a Long Short-Term Memory (LSTM) neural network, and independently composes brand new music sequences based on the acquired knowledge. 

## 🎵 Features

- **Custom Training Data**: Provide your own MIDI files to train the AI on any style or genre.
- **Deep Learning Architecture**: Utilizes an advanced LSTM neural network model written in PyTorch.
- **Automated Generation**: Produces unique, randomized compositions each time it executes.
- **Organized Storage**: Automatically saves the generated output with unique timestamps.

## 📁 Project Structure

- `setup/data`:
    - `midi_dataset/`: **Place your training MIDI files here.** This directory contains the original pieces the model learns from.
    - `generated_audios/`: **This is where your generated music goes!** After running the generation script, newly composed `.mid` files will be saved here with a unique timestamp (e.g. `generated_music_20260414_101621.mid`).
- `Core Scripts`:
    - `train.py`: Reads the MIDI files, builds note sequences, trains the LSTM model, and saves the learned network weights.
    - `generate.py`: Loads the trained weights and generates a brand new musical composition.
- `Generated Assets` (created by `train.py`):
    - `data_notes.pkl`: Stores the extracted sequence information from the original MIDI files.
    - `weights.pth`: Contains the PyTorch neural network knowledge weights.

## 🛠️ Installation & Setup

1. **Prerequisites**: Ensure you have Python installed on your system.
2. **Dependencies**: 
   Install the required libraries listed in the `requirements.txt` file and also PyTorch.
   ```bash
   pip install -r requirements.txt
   pip install torch
   ```

## 🚀 How to Run the Project

To use this application, follow these two main steps:

### Step 1: Train the AI Model

Before generating new music, the AI must learn from the given dataset. 
1. Make sure you have placed your `.mid` or `.midi` files inside the `midi_dataset` directory.
2. Run the training script:
   ```bash
   python train.py
   ```
   *Note: This process might take some time depending on your dataset size and hardware (CUDA/CPU).* 
   Once training completes, it will successfully establish the `data_notes.pkl` and `weights.pth` files.

### Step 2: Generate New Music

After the model finishes training, you can generate fresh, brand-new compositions.
1. Run the generation script:
   ```bash
   python generate.py
   ```
2. **Accessing the music**: The program will inform you when the generation is complete. You can find the newly created AI compositions saved automatically inside the **`generated_audios`** folder. Open these `.mid` files with any standard media player that supports MIDI to listen to the AI's composition.

And you can also listen the music via generated_music using Window Media Player Legacy inside the project folder

Enjoy the music!
