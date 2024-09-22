import argparse
import numpy as np
import torch
import librosa
import torchaudio
import torch.nn as nn
import matplotlib.pyplot as plt

from phn_ast.midi import save_midi
from phn_ast.decoding import FramewiseDecoder
from phn_ast.model import TranscriptionModel

ONSET_SCALE_FACTOR = 5
MIN_MIDI = 21
MAX_MIDI = 108
OUTPUT_FEATURES = 3 * (MAX_MIDI - MIN_MIDI + 1)

def infer(initial_model, model_file, input_file, output_file, pitch_sum, bpm, device):
    ckpt = torch.load(initial_model)
    config = ckpt['config']
    # model_state_dict = ckpt['model_state_dict']

    # model = TranscriptionModel(config)

    # model_size = model.combined_fc.in_features
    # model.combined_fc = nn.Linear(model_size, OUTPUT_FEATURES)

    # model.load_state_dict(model_state_dict)
    model = torch.load(model_file)
    model.to(device)

    model.to(device)
    model.eval()

    model.pitch_sum = pitch_sum

    decoder = FramewiseDecoder(config)

    audio, sr = torchaudio.load(input_file)
    audio = audio.numpy().T
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio_re = librosa.resample(audio, orig_sr=sr, target_sr=config['sample_rate'])
    audio_re = torch.from_numpy(audio_re).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(audio_re, None)[1]
        # p, i = decoder.decode(pred, audio=audio_re)

    plt.pcolor(pred)
    plt.savefig("inferred.png")

    # scale_factor = config['hop_length'] / config['sample_rate']

    # i = (np.array(i) * scale_factor).reshape(-1, 2)
    # p = np.array([round(midi) for midi in p])

    # save_midi(output_file, p, i, bpm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('initial_model', type=str)
    parser.add_argument('model_file', type=str)
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', nargs='?', default='out.mid', type=str)
    parser.add_argument('--pitch_sum', default='weighted_median', type=str)
    parser.add_argument('--bpm', '-b', default=120.0, type=float)
    parser.add_argument('--device', '-d',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    infer(args.initial_model, args.model_file, args.input_file, args.output_file, args.pitch_sum, args.bpm, args.device)
