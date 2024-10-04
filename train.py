import argparse
import numpy as np
import torch
import librosa
import torchaudio
import torch.nn as nn
from torch.utils.data.dataset import Dataset
# from sklearn.metrics import precision_score, recall_score

from phn_ast.midi import save_midi
from phn_ast.better_decoding import FramewiseDecoder
from phn_ast.model import TranscriptionModel
from phn_ast.feature import FeatureExtractor
import os

import matplotlib.pyplot as plt

import transformers
from subprocess import run
import mir_eval
from collections import defaultdict
import soundfile as sf
import resampy

ONSET_SCALE_FACTOR = 5
MIN_MIDI = 21
MAX_MIDI = 108
OUTPUT_FEATURES = 3 * (MAX_MIDI - MIN_MIDI + 2)

import torch.nn as nn
from torch.utils.data.dataset import Dataset

import mir_eval
from collections import defaultdict
import soundfile as sf
import resampy

class AudioDataset(Dataset):
    def __init__(self, config, data, labels) -> None:
        super().__init__()
        self.config = config
        self.data = data
        self.labels = labels
        self.paths = [os.path.join(root, file) for root, _, files in os.walk(data) for file in files]

    def get_labels(self, labels_path, path, length, start, end) -> np.ndarray:
        file, shift = os.path.split(path)[1].split("#")
        shift = int(shift.split(".")[0])
        # onset, offset, note, velocity, instrument
        notes = np.loadtxt(os.path.join(labels_path, f"{file}.tsv"), delimiter='\t', skiprows=1)
        notes[:,2] += shift

        start_t = start / self.config['sample_rate']
        end_t = end / self.config['sample_rate']
        start_note = len(notes)
        end_note = len(notes)
        for i, (on, off, _, _, _) in enumerate(notes):
            if start_note == len(notes) and on >= start_t:
                start_note = i
            if off >= end_t:
                end_note = i
                break
        notes = notes[start_note:end_note]
        notes[:,:2] = notes[:,:2] - start_t

        matrix = np.zeros((length, OUTPUT_FEATURES), dtype=np.float32)
        # matrix = np.zeros((length, 3), dtype=np.float32)
        for on, off, note, _, _ in notes:
            nt = int(note)
            if nt >= MIN_MIDI and nt <= MAX_MIDI:
                oni = int(on * self.config['sample_rate'] / self.config['hop_length'])
                offi = int(off * self.config['sample_rate'] / self.config['hop_length'])
                if oni < 0 or offi < 0 or oni >= len(matrix) or offi >= len(matrix):
                    print(f"WARN: note outside of label matrix range path: {path} oni: {oni}, offi: {offi} len(matrix): {len(matrix)}")
                    continue
                matrix[oni, (nt - MIN_MIDI) * 3] = 1
                matrix[offi, (nt - MIN_MIDI) * 3 + 1] = 1
                matrix[oni:offi,(nt - MIN_MIDI) * 3 + 2] = 1
                # matrix[oni, 0] = 1
                # matrix[offi, 1] = 1
                # matrix[oni:offi,2] = 1
        win = np.array([0.2, 0.6, 1, 0.6, 0.2])
        for note in range(0, matrix.shape[1] // 3 - 1):
            matrix[:,note * 3] = np.convolve(matrix[:,note * 3], win, mode='same')
            matrix[:,note * 3 + 1] = np.convolve(matrix[:,note * 3 + 1], win, mode='same')
        # matrix[:,0] = np.convolve(matrix[:,0], win, mode='same')
        # matrix[:,1] = np.convolve(matrix[:,1], win, mode='same')
        matrix[:,OUTPUT_FEATURES-3] = 1 - np.sum(matrix[:,:OUTPUT_FEATURES-3:3], axis=-1)
        matrix[:,OUTPUT_FEATURES-2] = 1 - np.sum(matrix[:,1:OUTPUT_FEATURES-3:3], axis=-1)
        matrix[:,OUTPUT_FEATURES-1] = 1 - np.sum(matrix[:,2:OUTPUT_FEATURES-3:3], axis=-1)
        return matrix, notes[:,:3]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        index, start, end = key
        path = self.paths[index]
        with sf.SoundFile(path) as f:
            start_rs = start * f.samplerate // self.config['sample_rate']
            end_rs = end * f.samplerate // self.config['sample_rate'] + 10
            f.seek(start_rs)
            data = librosa.to_mono(f.read(end_rs - start_rs).T)
            if f.samplerate != self.config['sample_rate']:
                data = resampy.resample(data, f.samplerate, self.config['sample_rate'])
            data = data[:end - start]
        fftlen = data.shape[0] // self.config['hop_length'] + 1
        labels, notes = self.get_labels(self.labels, path, fftlen, start, end)
        return data, labels, notes
    
    def get_len(self, index):
        path = self.paths[index]
        with sf.SoundFile(path) as f:
            f.seek(0, sf.SEEK_END)
            return f.tell() * self.config['sample_rate'] // f.samplerate

def eval_mean_square(x: np.ndarray) -> float:
    """
    Computes mean-square of x
    """
    return np.mean(x ** 2)


def power_to_db(x: float) -> float:
    """
    Computes 10log10(x)
    """
    return 10 * np.log10(x + 1e-20)


def eval_rms_db(x: np.ndarray) -> float:
    """
    Computes rms-fs of x
    """
    rms_square_raw = eval_mean_square(x)
    rms_db = power_to_db(rms_square_raw)
    return rms_db

import random
class SignalSampler:
    def __init__(
        self,
        config,
        dataset,
        len=None,
        crop_size_sec: float = 5.0,
        min_rms_db: float | None = -38,
        det=False,
    ) -> None:
        """
        paths: list of absolute paths to the files we are going to sample from
        crop_size_sec: the size of generated chunks, in seconds
        min_rms_db: chunks with RMS lower than this should be discarded
        sr: samplerate
        """
        self.config = config
        self.dataset = dataset
        self.crop_size_ticks = int(crop_size_sec * self.config['sample_rate'])
        self.min_rms_db = min_rms_db
        self.sr = self.config['sample_rate']
        self.len = len
        self.det = det

    def _sample_from_single_file(
        self, crop_size_ticks: int | None = None
    ) -> np.ndarray:
        """
        Reads a random crop of size crop_size_frames from path.
        If the file is shorter, reads the full file.
        """
        if crop_size_ticks is None:
            crop_size_ticks = self.crop_size_ticks
        crop_size_frames = crop_size_ticks // self.config["hop_length"] + 1
        audio_idx = random.randint(0, len(self.dataset) - 1)
        file_len = self.dataset.get_len(audio_idx)
        file_duration_frames = file_len // self.config["hop_length"] + 1
        if file_len < crop_size_ticks:
            audio, labels, notes = self.dataset[audio_idx, 0, file_len]
            assert len(audio) == file_len
            assert len(labels) == file_duration_frames
            audio = np.pad(audio, (0, file_len - audio.shape[0]))
            labels = np.pad(labels, ((0, crop_size_frames - file_duration_frames), (0, 0)))
        start = random.randint(0, file_len - crop_size_ticks)
        end = start + crop_size_ticks
        audio, labels, notes = self.dataset[audio_idx, start, end]
        assert len(audio) == crop_size_ticks
        assert len(labels) == crop_size_frames
        return audio, labels, notes

    def __len__(self) -> int:
        return len(self.dataset) * 60 if self.len is None else self.len

    def __getitem__(self, index) -> np.ndarray:
        """
        Generates a chunk of audio data of length self.crop_size_frames.

        1. Samples a random file from self.paths

        2. Reads its random crop of the target size (initialized as self.crop_size_frames).
           If the file is shorter, reads the full file.
           <this should be done in self._sample_from_single_file>

        3. Checks RMS of the crop.
           The crop is discarded if its rms is lower than self.min_rms_db.
           Otherwise it is accumulated

        4. Returns the concatenation of accumulated crops
           if their total length reaches self.crop_size_frames.
           Otherwise sets target size (for 2) to n_frames_remaining and repeats 1-4
        """
        if self.det:
            random.seed(index)
        else:
            random.seed()
        while True:
            audio, label, notes = self._sample_from_single_file(self.crop_size_ticks)
            if self.min_rms_db is not None:
                chunk_rms_db = eval_rms_db(audio)
                if chunk_rms_db < self.min_rms_db or len(notes) == 0:
                    continue
            break
        notes = notes[:50]
        notes = np.pad(notes, ((0, 50 - len(notes)), (0, 0)), constant_values=-1)

        crop_size_frames = self.crop_size_ticks // self.config["hop_length"] + 1
        assert audio.ndim == 1, audio.shape
        assert audio.shape[0] == self.crop_size_ticks, audio.shape
        assert label.ndim == 2, label.shape
        assert label.shape[0] == crop_size_frames, label.shape
        assert notes.shape[0] == 50, notes.shape
        assert notes.shape[1] == 3, notes.shape

        return dict(x=audio.astype('float32'), labels=label.astype('float32'), notes=notes.astype('float32'))

class S3Callback(transformers.TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        model_path = os.path.join("run", f"model-pitch{state.epoch}.pt")
        torch.save(model.state_dict(), model_path)
        run(["aws", "s3", "cp", model_path, "s3://chp"])

def make_compute_metrics(config):
    decoder = FramewiseDecoder(config)
    def compute_metrics(eval_prediction):
        preds = eval_prediction.predictions[0]
        # audio = eval_prediction.predictions[1]
        notes = eval_prediction.predictions[1]
        metrics = []
        for pred, n in zip(preds, notes):
            i, p = decoder.decode(pred)
            p = np.array([round(midi + MIN_MIDI) for midi in p])
            # Remove padding
            end_nt = len(n)
            for j, (on, off, note) in enumerate(n):
                if on == -1:
                    end_nt = j
                    break
            n = n[:end_nt]
            if len(n) == 0:
                print("WARN: frame without notes in batch")
                continue
            p = np.clip(p, MIN_MIDI, MAX_MIDI)
            metrics.append(mir_eval.transcription.evaluate(n[:,:2], librosa.midi_to_hz(n[:,2]), i, librosa.midi_to_hz(p)))
        avg_metrics = defaultdict(int)
        for b in metrics:
            for k, v in b.items():
                avg_metrics[k] += v
        for k in avg_metrics:
            avg_metrics[k] /= len(metrics)
        return dict(avg_metrics)
    return compute_metrics

#     onset_precision = precision_score(labels[...,::3].reshape(-1), preds[...,::3].reshape(-1))
#     offset_precision = precision_score(labels[...,1::3].reshape(-1), preds[...,1::3].reshape(-1))
#     frame_precision = precision_score(labels[...,2::3].reshape(-1), preds[...,2::3].reshape(-1))
#     onset_recall = recall_score(labels[...,::3].reshape(-1), preds[...,::3].reshape(-1))
#     offset_recall = recall_score(labels[...,1::3].reshape(-1), preds[...,1::3].reshape(-1))
#     frame_recall = recall_score(labels[...,2::3].reshape(-1), preds[...,2::3].reshape(-1))

#     return dict(
#         onset_precision=onset_precision,
#         offset_precision=offset_precision,
#         frame_precision=frame_precision,
#         onset_recall=onset_recall,
#         offset_recall=offset_recall,
#         frame_recall=frame_recall
#     )

def train(model_file, train, eval, run, device):
    ckpt = torch.load(model_file)
    config = ckpt['config']
    model_state_dict = ckpt['model_state_dict']

    model = TranscriptionModel(config)
    model.load_state_dict(model_state_dict)
    ...
    model_size = model.combined_fc.in_features
    model.combined_fc = nn.Linear(model_size, OUTPUT_FEATURES)
    # model.load_state_dict(torch.load("run/model14.0.pt"))
    # for p in model.parameters():
    #     p.requires_grad=False
    
    # for p in model.combined_fc.parameters():
    #     p.requires_grad = True

    traind = SignalSampler(config, AudioDataset(config, "train", "labels/train"), len=2**13, min_rms_db=None)
    evald = SignalSampler(config, AudioDataset(config, "test", "labels/train"), len=1024, det=True)

    ta = transformers.TrainingArguments(output_dir="out", evaluation_strategy="epoch", per_device_train_batch_size=64, per_device_eval_batch_size=64, num_train_epochs=100, report_to="wandb")
    trainer = transformers.Trainer(
        model, args=ta, train_dataset=traind, eval_dataset=evald, compute_metrics=make_compute_metrics(config))
    trainer.add_callback(S3Callback())
    print(trainer.evaluate())
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('train', type=str, default="train")
    parser.add_argument('eval', type=str, default="test")
    parser.add_argument('run', type=str, default="run")
    parser.add_argument('--bpm', '-b', default=120.0, type=float)
    parser.add_argument('--device', '-d',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    train(args.model_file, args.train, args.eval, args.run, args.device)
