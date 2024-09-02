""" TF2 support by compatibility """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MusicModel(nn.Module):
    def __init__(self, input_notes_shape, input_offsets_shape, input_durations_shape, n_vocab_notes, n_vocab_offsets, n_vocab_durations):
        super(MusicModel, self).__init__()
        
        # Note branch
        self.lstm_notes = nn.LSTM(input_size=input_notes_shape[2], hidden_size=256, num_layers=1, batch_first=True, dropout=0)
        
        # Offset branch
        self.lstm_offsets = nn.LSTM(input_size=input_offsets_shape[2], hidden_size=256, num_layers=1, batch_first=True, dropout=0)
        
        # Duration branch
        self.lstm_durations = nn.LSTM(input_size=input_durations_shape[2], hidden_size=256, num_layers=1, batch_first=True, dropout=0)
        
        # Combined branch
        self.lstm_combined_1 = nn.LSTM(input_size=256*3, hidden_size=512, num_layers=1, batch_first=True, dropout=0)
        self.lstm_combined_2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0)
        self.batch_norm = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 256)
        
        # Output branches
        self.fc_notes = nn.Linear(256, 128)
        self.fc_notes_out = nn.Linear(128, n_vocab_notes)
        
        self.fc_offsets = nn.Linear(256, 128)
        self.fc_offsets_out = nn.Linear(128, n_vocab_offsets)
        
        self.fc_durations = nn.Linear(256, 128)
        self.fc_durations_out = nn.Linear(128, n_vocab_durations)
        
    def forward(self, notes, offsets, durations):
        # Note branch
        notes, _ = self.lstm_notes(notes)
        notes = F.dropout(notes, 0.2)
        
        # Offset branch
        offsets, _ = self.lstm_offsets(offsets)
        offsets = F.dropout(offsets, 0.2)
        
        # Duration branch
        durations, _ = self.lstm_durations(durations)
        durations = F.dropout(durations, 0.2)
        
        # Concatenate branches
        combined = torch.cat((notes, offsets, durations), dim=2)
        
        # Combined branch
        combined, _ = self.lstm_combined_1(combined)
        combined = F.dropout(combined, 0.3)
        combined, _ = self.lstm_combined_2(combined)
        combined = combined[:, -1, :]  # Take the last output
        combined = self.batch_norm(combined)
        combined = F.dropout(combined, 0.3)
        combined = F.relu(self.fc1(combined))
        
        # Output branches
        notes_out = F.relu(self.fc_notes(combined))
        notes_out = self.fc_notes_out(notes_out)
        
        offsets_out = F.relu(self.fc_offsets(combined))
        offsets_out = self.fc_offsets_out(offsets_out)
        
        durations_out = F.relu(self.fc_durations(combined))
        durations_out = self.fc_durations_out(durations_out)
        
        return notes_out, offsets_out, durations_out
	
def train_network():
	""" Train a Neural Network to generate music """
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("RUNNING ON DEVICE ", device)
	notes, offsets, durations = get_notes()

	# Prepare notes 
	n_vocab_notes = len(set(notes))
	network_input_notes, network_output_notes = prepare_sequences(notes, n_vocab_notes)
	
	# Prepare notes 
	n_vocab_offsets = len(set(offsets))
	network_input_offsets, network_output_offsets = prepare_sequences(offsets, n_vocab_offsets)
	
	# Prepare notes 
	n_vocab_durations = len(set(durations))
	network_input_durations, network_output_durations = prepare_sequences(durations, n_vocab_durations)

	# Convert to PyTorch tensors
	network_input_notes = torch.tensor(network_input_notes, dtype=torch.float32).to(device)
	network_output_notes = (network_output_notes.clone().detach().long()).to(device)
	network_input_offsets = torch.tensor(network_input_offsets, dtype=torch.float32).to(device)
	network_output_offsets = (network_output_offsets.clone().detach().long()).to(device)
	network_input_durations = torch.tensor(network_input_durations, dtype=torch.float32).to(device)
	network_output_durations = (network_output_durations.clone().detach().long()).to(device)

	# Create Dataloader
	dataset = TensorDataset(network_input_notes, network_input_offsets, network_input_durations, network_output_notes, network_output_offsets, network_output_durations)
	dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
	
	
	model = MusicModel(
        input_notes_shape=network_input_notes.shape,
        input_offsets_shape=network_input_offsets.shape,
        input_durations_shape=network_input_durations.shape,
        n_vocab_notes=n_vocab_notes,
        n_vocab_offsets=n_vocab_offsets,
        n_vocab_durations=n_vocab_durations
    ).to(device)
	print(network_input_notes.shape,
        network_input_offsets.shape,
        network_input_durations.shape,
        n_vocab_notes,
        n_vocab_offsets,
        n_vocab_durations)
	
	# Define Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	# Training loop
	model.train()
	epochs = 500
	model_name= input("New trained Model name:")
	best_loss = None
	for epoch in range(epochs):
		for batch in dataloader:
			notes_in, offsets_in, durations_in, notes_out, offsets_out, durations_out = batch
			# Move batch data to GPU
			notes_in, offsets_in, durations_in = notes_in.to(device), offsets_in.to(device), durations_in.to(device)
			notes_out, offsets_out, durations_out = notes_out.to(device), offsets_out.to(device), durations_out.to(device)
			optimizer.zero_grad()
			notes_pred, offsets_pred, durations_pred = model(notes_in, offsets_in, durations_in)			
			loss_notes = criterion(notes_pred, notes_out)			
			loss_offsets = criterion(offsets_pred, offsets_out)
			loss_durations = criterion(durations_pred, durations_out)
			loss = loss_notes + loss_offsets + loss_durations
			loss.backward()
			optimizer.step()
			
		print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
		if(best_loss == None):
			torch.save(model.state_dict(), "models/"+ model_name +'.pth')
		elif best_loss > loss.item():
			torch.save(model.state_dict(), "models/"+model_name +'.pth')
    	
	

def get_notes():
	""" Get all the notes and chords from the midi files """
	notes = []
	offsets = []
	durations = []
	
	midi_files_directory = "vgame-piano/*.mid"

	for file in glob.glob(midi_files_directory):
		midi = converter.parse(file)

		print("Parsing %s" % file)

		notes_to_parse = None

		try: # file has instrument parts
			s2 = instrument.partitionByInstrument(midi)
			notes_to_parse = s2.parts[0].recurse() 
		except: # file has notes in a flat structure
			notes_to_parse = midi.flat.notes
		
		
		offsetBase = 0
		for element in notes_to_parse:
			isNoteOrChord = False
			
			if isinstance(element, note.Note):
				notes.append(str(element.pitch))
				isNoteOrChord = True
			elif isinstance(element, chord.Chord):
				notes.append('.'.join(str(n) for n in element.normalOrder))
				isNoteOrChord = True
			
			if isNoteOrChord:				
				offsets.append(str(element.offset - offsetBase))				
				durations.append(str(element.duration.quarterLength))
				
				isNoteOrChord = False
				offsetBase = element.offset
				

	with open('data/notes', 'wb') as filepath:
		pickle.dump(notes, filepath)
	
	with open('data/durations', 'wb') as filepath:
		pickle.dump(durations, filepath)
		
	with open('data/offsets', 'wb') as filepath:
		pickle.dump(offsets, filepath)
	
	return notes, offsets, durations

def prepare_sequences(notes, n_vocab):
	""" Prepare the sequences used by the Neural Network """
	sequence_length = 100

	# get all pitch names
	pitchnames = sorted(set(item for item in notes))

	 # create a dictionary to map pitches to integers
	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

	network_input = []
	network_output = []

	# create input sequences and the corresponding outputs
	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		sequence_out = notes[i + sequence_length]
		network_input.append([note_to_int[char] for char in sequence_in])
		network_output.append(note_to_int[sequence_out])

	n_patterns = len(network_input)

	# reshape the input into a format compatible with LSTM layers
	network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
	# normalize input
	network_input = network_input / float(n_vocab)
	network_input = torch.tensor(network_input, dtype=torch.float32)

	network_output = torch.tensor(network_output, dtype=torch.long)

	return (network_input, network_output)

if __name__ == '__main__':	
	train_network()
