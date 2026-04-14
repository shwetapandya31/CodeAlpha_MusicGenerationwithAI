import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, corpus
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MusicModel(nn.Module):
    def __init__(self, n_vocab):
        super(MusicModel, self).__init__()
        # PyTorch LSTM expects input of shape (batch, seq_len, features) if batch_first=True
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
        # We only need the output of the last sequence step
        x = x[:, -1, :]
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x

def get_notes():
    """ Get all the notes and chords from the local midi_dataset folder """
    notes = []
    
    paths = glob.glob("midi_dataset/*.mid") + glob.glob("midi_dataset/*.midi")
    if not paths:
        print("No MIDI files found in 'midi_dataset' folder. Please ensure they end in .mid")
        return []
        
    print(f"Found {len(paths)} custom MIDI pieces. Processing...")
    
    for i, file in enumerate(paths):
        # We use converter.parse instead of corpus.parse for local files
        midi = converter.parse(file)
        print(f"Parsing file {i+1}...")

        notes_to_parse = None
        try:
            s2 = instrument.partitionByInstrument(midi)
            if s2:
                notes_to_parse = s2.parts[0].recurse() 
            else:
                notes_to_parse = midi.flat.notes
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data_notes.pkl', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 50
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)

    # Convert to PyTorch tensors
    tensor_in = torch.tensor(network_input, dtype=torch.float32)
    tensor_out = torch.tensor(network_output, dtype=torch.long) # CrossEntropyLoss expects class indices

    return tensor_in, tensor_out

def train(model, tensor_in, tensor_out):
    """ train the neural network """
    print("Starting training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    dataset = TensorDataset(tensor_in, tensor_out)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    epochs = 5
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_in, batch_out in dataloader:
            batch_in, batch_out = batch_in.to(device), batch_out.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_in)
            loss = criterion(outputs, batch_out)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'weights.pth')
            print("Saved best model.")

if __name__ == '__main__':
    print("Gathering notes...")
    notes = get_notes()
    
    n_vocab = len(set(notes))
    print(f"Unique notes/chords (vocab size): {n_vocab}")
    
    tensor_in, tensor_out = prepare_sequences(notes, n_vocab)
    
    print("Building model...")
    model = MusicModel(n_vocab)
    
    train(model, tensor_in, tensor_out)
    print("Training complete! Model weights saved as weights.pth")
