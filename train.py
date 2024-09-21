import argparse
import numpy as np
import torch
import librosa
import torchaudio
import torch.nn as nn
from torch.utils.data.dataset import Dataset

from phn_ast.midi import save_midi
from phn_ast.decoding import FramewiseDecoder
from phn_ast.model import TranscriptionModel
import os

ONSET_SCALE_FACTOR = 5
MIN_MIDI = 21
MAX_MIDI = 108
OUTPUT_FEATURES = 3 * (MAX_MIDI - MIN_MIDI + 1)

class AudioDataset(Dataset):
    def __init__(self, config, data, labels) -> None:
        super().__init__()
        self.config = config
        self.data = data
        self.labels = labels
        self.paths = [os.path.join(root, files) for root, _, files in os.walk(data) for file in files]
        
    def get_labels(self, labels, path, length) -> np.ndarray:
        file, shift = os.path.split(path)[1].split("#")
        shift = int(shift.split(".")[0])
        # onset, offset, note, velocity, instrument
        labels = np.loadtxt(os.path.join(labels, f"{file}.tsv"), delimiter='\t', skiprows=1)
        labels[:,:2] += shift
        matrix = np.zeros(length, OUTPUT_FEATURES, dtype=np.float32)
        for on, off, note, _, _ in labels:
            if note >= MIN_MIDI and note <= MAX_MIDI:
               oni = int(on * self.config['sample_rate'] / self.config['hop_length'])
               offi = int(off * self.config['sample_rate'] / self.config['hop_length'])
               matrix[oni, (note - MIN_MIDI) * 3] = 1
               matrix[offi, (note - MIN_MIDI) * 3 + 1] = 1
               matrix[oni:offi+1,(note - MIN_MIDI) * 3 + 2] = 1
        win = librosa.filters.get_window('triangle', ONSET_SCALE_FACTOR)
        win /= np.max(win)
        for note in range(0, matrix.shape[1] // 3):
            matrix[note * 3] = np.convolve(matrix[note * 3], win, mode='same')
            matrix[note * 3 + 1] = np.convolve(matrix[note * 3 + 1], win, mode='same')
        return matrix

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index) -> Any:
        path = self.paths[index]
        audio = librosa.load(path, sr=self.config['sample_rate'])
        fftlen = (audio.shape[0] + self.config['hop_length'] - 1) / self.config['hop_length']
        labels = self.get_labels(self.labels, path, fftlen)
        return path, labels


def train(model_file, train, eval, run, device):
    ckpt = torch.load(model_file)
    config = ckpt['config']
    model_state_dict = ckpt['model_state_dict']

    model = TranscriptionModel(config)
    model.load_state_dict(model_state_dict)
    model.to(device)

    model_size = model.combined_fc.in_features
    model.combined_fc = nn.Linear(model_size, OUTPUT_FEATURES)

    model.pitch_sum = pitch_sum

    decoder = FramewiseDecoder(config)

    audio, sr = torchaudio.load(input_file)
    audio = audio.numpy().T
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio_re = librosa.resample(audio, orig_sr=sr, target_sr=config['sample_rate'])
    audio_re = torch.from_numpy(audio_re).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(audio_re)
        p, i = decoder.decode(pred, audio=audio_re)

    scale_factor = config['hop_length'] / config['sample_rate']

    i = (np.array(i) * scale_factor).reshape(-1, 2)
    p = np.array([round(midi) for midi in p])

    save_midi(output_file, p, i, bpm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('train', type=str)
    parser.add_argument('eval', type=str)
    parser.add_argument('run', type=str)
    parser.add_argument('--bpm', '-b', default=120.0, type=float)
    parser.add_argument('--device', '-d',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    train(args.model_file, args.train, args.eval, args.run, args.device)
