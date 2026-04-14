import pickle
import numpy as np
from music21 import instrument, note, stream, chord
import torch
import torch.nn as nn
import os
from datetime import datetime

class MusicModel(nn.Module):
    def __init__(self, n_vocab):
        super(MusicModel, self).__init__()
        self.lstm1 = nn.LSTM(1, 256, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(256, 256, batch_first=True)
        self.dense1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(128, n_vocab)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 50
    network_input = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    n_patterns = len(network_input)
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    normalized_input = normalized_input / float(n_vocab)

    return network_input, normalized_input

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    start = np.random.randint(0, len(network_input)-1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    print("Generating 200 notes/chords...")
    
    with torch.no_grad():
        for note_index in range(200):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            
            tensor_in = torch.tensor(prediction_input, dtype=torch.float32).to(device)
            prediction = model(tensor_in)
            
            # Get the probabilities with temperature for creativity
            temperature = 1.2
            probabilities = torch.softmax(prediction / temperature, dim=1)
            index = torch.multinomial(probabilities, 1).item()
            
            result = int_to_note[index]
            prediction_output.append(result)

            pattern.append(index)
            pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file """
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Saxophone()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Saxophone()
            output_notes.append(new_note)

        offset += 0.5

    # Create output directory
    output_dir = 'generated_audios'
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f'generated_music_{timestamp}.mid')

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)
    print(f"Saved {filename} successfully!")

if __name__ == '__main__':
    try:
        with open('data_notes.pkl', 'rb') as filepath:
            notes = pickle.load(filepath)
    except FileNotFoundError:
        print("data_notes.pkl not found. Please run train.py first.")
        exit(1)

    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))

    print("Preparing sequences...")
    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicModel(n_vocab)
    try:
        model.load_state_dict(torch.load('weights.pth', map_location=device))
    except Exception as e:
        print("Could not load weights.pth. Did you run train.py first?")
        raise e
    
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)
