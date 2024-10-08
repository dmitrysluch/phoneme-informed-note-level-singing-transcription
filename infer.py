import argparse
import numpy as np
import torch
import librosa
import torchaudio
import torch.nn as nn
import matplotlib.pyplot as plt

from phn_ast.midi import save_midi
from phn_ast.better_decoding import FramewiseDecoder
from phn_ast.model import TranscriptionModel

ONSET_SCALE_FACTOR = 5
MIN_MIDI = 21
MAX_MIDI = 108
OUTPUT_FEATURES = 3 * (MAX_MIDI - MIN_MIDI + 2)

def infer(initial_model, model_file, input_file, output_file, pitch_sum, bpm, device):
    ckpt = torch.load(initial_model)
    config = ckpt['config']
    model_state_dict = ckpt['model_state_dict']

    model = TranscriptionModel(config)

    model_size = model.combined_fc.in_features
    model.combined_fc = nn.Linear(model_size, OUTPUT_FEATURES)

    # model.load_state_dict(model_state_dict)
    model.load_state_dict(torch.load("model-pitch15.0.pt")) # B - моя
    model = model.to(device)
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
        pred = model(audio_re, None, None)[1].squeeze(0)
        i, p = decoder.decode(pred.cpu().detach().numpy())
    
    # print(p.shape, i.shape)
    # torch.save(pred.detach().cpu(), "pred.pt")
    # plt.pcolor(pred.detach().cpu().numpy()[0,1000:2000])
    # plt.savefig("inferred.png")
    p = np.array([round(midi + MIN_MIDI) for midi in p])

    print(i)
    print(p)

    # for (f, t), p in zip(i, p):
    #     plt.plot([f, t], [p, p], 'r')

    # plt.savefig("pred.png")
    # plt.clf()

    save_midi(output_file, p, i, bpm=120, add_start_point=True)


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
