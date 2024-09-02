""" This module generates notes for a midi file using the
	trained neural network """
import pickle
import numpy
from music21 import stream, instrument, note, chord

import numpy as np
import torch
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

def load_model(input_notes_shape, input_offsets_shape, input_durations_shape, n_vocab_notes, n_vocab_offsets, n_vocab_durations):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = MusicModel(
        input_notes_shape=input_notes_shape,
        input_offsets_shape=input_offsets_shape,
        input_durations_shape=input_durations_shape,
        n_vocab_notes=n_vocab_notes,
        n_vocab_offsets=n_vocab_offsets,
        n_vocab_durations=n_vocab_durations
    )
	model_path = 'models/vmusic_model.pth'
	model.load_state_dict(torch.load(model_path))
	model.to(device)
	model.eval()  # Set the model to evaluation mode
	return model

def generate():
	""" Generate a piano midi file """
	#load the notes used to train the model
	with open('data/notes', 'rb') as filepath:
		notes = pickle.load(filepath)
	
	with open('data/durations', 'rb') as filepath:
		durations = pickle.load(filepath)
	
	with open('data/offsets', 'rb') as filepath:
		offsets = pickle.load(filepath)

	notenames = sorted(set(item for item in notes))
	n_vocab_notes = len(set(notes))
	network_input_notes, normalized_input_notes = prepare_sequences(notes, notenames, n_vocab_notes)
	
	offsetnames = sorted(set(item for item in offsets))
	n_vocab_offsets = len(set(offsets))
	network_input_offsets, normalized_input_offsets = prepare_sequences(offsets, offsetnames, n_vocab_offsets)
	
	durationames = sorted(set(item for item in durations))
	n_vocab_durations = len(set(durations))
	network_input_durations, normalized_input_durations = prepare_sequences(durations, durationames, n_vocab_durations)
	
	# the sizes may vary, check the sizes of model when training
	model = load_model(torch.Size([9067, 100, 1]), torch.Size([9067, 100, 1]), torch.Size([9067, 100, 1]), 501, 147, 50)
		
	prediction_output = generate_notes(model, network_input_notes, network_input_offsets, network_input_durations, notenames, offsetnames, durationames, n_vocab_notes, n_vocab_offsets, n_vocab_durations)
	create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
	""" Prepare the sequences used by the Neural Network """
	# map between notes and integers and back
	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

	sequence_length = 100
	network_input = []
	output = []
	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		sequence_out = notes[i + sequence_length]
		network_input.append([note_to_int[char] for char in sequence_in])
		output.append(note_to_int[sequence_out])

	n_patterns = len(network_input)

	# reshape the input into a format compatible with LSTM layers
	normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
	# normalize input
	normalized_input = normalized_input / float(n_vocab)

	return (network_input, normalized_input)

def generate_notes(model, network_input_notes, network_input_offsets, network_input_durations, notenames, offsetnames, durationames, n_vocab_notes, n_vocab_offsets, n_vocab_durations):
	""" Generate notes from the neural network based on a sequence of notes """
	# pick a random sequence from the input as a starting point for the prediction
	start = numpy.random.randint(0, len(network_input_notes)-1)
	start2 = numpy.random.randint(0, len(network_input_offsets)-1)
	start3 = numpy.random.randint(0, len(network_input_durations)-1)

	int_to_note = dict((number, note) for number, note in enumerate(notenames))
	print(int_to_note)
	int_to_offset = dict((number, note) for number, note in enumerate(offsetnames))
	int_to_duration = dict((number, note) for number, note in enumerate(durationames))

	pattern = network_input_notes[start]
	pattern2 = network_input_offsets[start2]
	pattern3 = network_input_durations[start3]
	prediction_output = []


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# generate notes or chords
	for note_index in range(120):
		note_prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
		predictedNote = note_prediction_input[-1][-1][-1]	
		
		note_prediction_input = note_prediction_input / float(n_vocab_notes)
		note_prediction_input = torch.tensor(note_prediction_input, dtype=torch.float32).to(device)
		
		offset_prediction_input = numpy.reshape(pattern2, (1, len(pattern2), 1))
		offset_prediction_input = offset_prediction_input / float(n_vocab_offsets)
		offset_prediction_input = torch.tensor(offset_prediction_input, dtype=torch.float32).to(device)
		
		duration_prediction_input = numpy.reshape(pattern3, (1, len(pattern3), 1))
		duration_prediction_input = duration_prediction_input / float(n_vocab_durations)
		duration_prediction_input = torch.tensor(duration_prediction_input, dtype=torch.float32).to(device)

		with torch.no_grad():
			prediction = model(note_prediction_input, offset_prediction_input, duration_prediction_input)

		index = torch.argmax(prediction[0]).item()
		#print(index)
		result = int_to_note[index]
		#print(result)
		
		offset = torch.argmax(prediction[1]).item()
		offset_result = int_to_offset[offset]
		#print("offset")
		#print(offset_result)
		
		duration = torch.argmax(prediction[2]).item()
		duration_result = int_to_duration[duration]
		#print("duration")
		#print(duration_result)
		
		print(f"Next note: {int_to_note[predictedNote]} - Duration: {int_to_duration[duration]} - Offset: {int_to_offset[offset]}")
		
		
		#
		prediction_output.append([result, offset_result, duration_result])

		pattern = np.append(pattern, index)[1:]
		pattern2 = np.append(pattern2, offset)[1:]
		pattern3 = np.append(pattern3, duration)[1:]

	return prediction_output

def create_midi(prediction_output_all):
	""" convert the output from the prediction to notes and create a midi file
		from the notes """
	offset = 0
	output_notes = []
	
	#prediction_output = prediction_output_all
	
	offsets = []
	durations = []
	notes = []
	
	for x in prediction_output_all:
		print(x)
		notes = numpy.append(notes, x[0])
		try:
			offsets = numpy.append(offsets, float(x[1]))
		except:
			num, denom = x[1].split('/')
			x[1] = float(num)/float(denom)
			offsets = numpy.append(offsets, float(x[1]))
			
		durations = numpy.append(durations, x[2])
	
	print("---")
	print(notes)
	print(offsets)
	print(durations)

	# create note and chord objects based on the values generated by the model
	x = 0 # this is the counter
	for pattern in notes:
		# pattern is a chord
		if ('.' in pattern) or pattern.isdigit():
			notes_in_chord = pattern.split('.')
			notes = []
			for current_note in notes_in_chord:
				new_note = note.Note(int(current_note))
				new_note.storedInstrument = instrument.Piano()
				notes.append(new_note)
			new_chord = chord.Chord(notes)
			
			try:
				new_chord.duration.quarterLength = float(durations[x])
			except:
				num, denom = durations[x].split('/')
				new_chord.duration.quarterLength = float(num)/float(denom)
			
			new_chord.offset = offset
			
			output_notes.append(new_chord)
		# pattern is a note
		else:
			new_note = note.Note(pattern)
			new_note.offset = offset
			new_note.storedInstrument = instrument.Piano()
			try:
				new_note.duration.quarterLength = float(durations[x])
			except:
				num, denom = durations[x].split('/')
				new_note.duration.quarterLength = float(num)/float(denom)
			
			output_notes.append(new_note)

		# increase offset each iteration so that notes do not stack
		try:
			offset += offsets[x]
		except:
			num, denom = offsets[x].split('/')
			offset += num/denom
				
		x = x+1

	midi_stream = stream.Stream(output_notes)
	song_name = input("New song name:")
	midi_stream.write('midi', fp= song_name +'.mid')

if __name__ == '__main__':
	generate()
