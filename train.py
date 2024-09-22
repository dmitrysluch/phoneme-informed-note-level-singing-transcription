import argparse
import numpy as np
import torch
import librosa
import torchaudio
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from sklearn.metrics import precision_score, recall_score

from phn_ast.midi import save_midi
from phn_ast.decoding import FramewiseDecoder
from phn_ast.model import TranscriptionModel
from phn_ast.feature import FeatureExtractor
import os

import matplotlib.pyplot as plt

import transformers
from subprocess import run


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
        self.paths = [os.path.join(root, file) for root, _, files in os.walk(data) for file in files]

    def get_labels(self, labels, path, length) -> np.ndarray:
        file, shift = os.path.split(path)[1].split("#")
        shift = int(shift.split(".")[0])
        # onset, offset, note, velocity, instrument
        labels = np.loadtxt(os.path.join(labels, f"{file}.tsv"), delimiter='\t', skiprows=1)
        labels[:,2] += shift
        matrix = np.zeros((length, OUTPUT_FEATURES), dtype=np.float32)
        for on, off, note, _, _ in labels:
            nt = int(note)
            if nt >= MIN_MIDI and nt <= MAX_MIDI:
               oni = int(on * self.config['sample_rate'] / self.config['hop_length'])
               offi = int(off * self.config['sample_rate'] / self.config['hop_length'])
               matrix[oni, (nt - MIN_MIDI) * 3] = 1
               matrix[offi, (nt - MIN_MIDI) * 3 + 1] = 1
               matrix[oni:offi,(nt - MIN_MIDI) * 3 + 2] = 1
        win = np.array([0.2, 0.6, 1, 0.6, 0.2])
        for note in range(0, matrix.shape[1] // 3):
            matrix[:,note * 3] = np.convolve(matrix[:,note * 3], win, mode='same')
            matrix[:,note * 3 + 1] = np.convolve(matrix[:,note * 3 + 1], win, mode='same')
        return matrix

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        audio, sr = librosa.load(path, sr=self.config['sample_rate'])
        assert sr == self.config['sample_rate']
        fftlen = (audio.shape[0] + self.config['win_length']) // self.config['hop_length']
        labels = self.get_labels(self.labels, path, fftlen)
        return audio, labels

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
    ) -> None:
        """
        paths: list of absolute paths to the files we are going to sample from
        crop_size_sec: the size of generated chunks, in seconds
        min_rms_db: chunks with RMS lower than this should be discarded
        sr: samplerate
        """
        self.config = config
        self.dataset = dataset
        self.crop_size_frames = int(crop_size_sec * self.config['sample_rate'] / self.config['hop_length'])
        self.min_rms_db = min_rms_db
        self.sr = self.config['sample_rate']
        self.len = len

    def _sample_from_single_file(
        self, crop_size_frames: int | None = None
    ) -> np.ndarray:
        """
        Reads a random crop of size crop_size_frames from path.
        If the file is shorter, reads the full file.
        """
        if crop_size_frames is None:
            crop_size_frames = self.crop_size_frames
        audio, labels = self.dataset[random.randint(0, len(self.dataset) - 1)]
        file_duration_frames = len(labels)
        if file_duration_frames < crop_size_frames:
            return audio, labels
        start = random.randint(0, file_duration_frames - crop_size_frames - 1)
        labels = labels[start:start+crop_size_frames]
        start_a, end_a = librosa.frames_to_samples([start, start+crop_size_frames], hop_length=self.config["hop_length"])
        audio = audio[start_a: end_a]
        return audio, labels

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
        audio_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        duration_frames_remaining = self.crop_size_frames
        while duration_frames_remaining > 0:
            audio, label = self._sample_from_single_file(duration_frames_remaining)
            if self.min_rms_db is not None:
                chunk_rms_db = eval_rms_db(audio)
                if chunk_rms_db < self.min_rms_db:
                    continue
            audio_chunks.append(audio)
            label_chunks.append(label)
            duration_frames_remaining -= label.shape[0]
        audio_res = np.concatenate(audio_chunks)
        label_res = np.concatenate(label_chunks)

        assert audio_res.ndim == 1, result.shape
        assert label_res.ndim == 2, result.shape
        assert label_res.shape[0] == self.crop_size_frames

        return dict(x=audio_res, labels=label_res)

class S3Callback(transformers.TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        model_path = os.path.join("run", f"model{state.epoch}.pt")
        torch.save(model.state_dict(), model_path)
        run(["aws", "s3", "cp", model_path, "s3://chp"])

def compute_metrics(eval_prediction):
    preds = eval_prediction.predictions[0]
    preds = (preds > 0.5)
    labels = eval_prediction.predictions[1]
    labels = (labels > 0.9999)
    onset_precision = precision_score(labels[...,::3].reshape(-1), preds[...,::3].reshape(-1))
    offset_precision = precision_score(labels[...,1::3].reshape(-1), preds[...,1::3].reshape(-1))
    frame_precision = precision_score(labels[...,2::3].reshape(-1), preds[...,2::3].reshape(-1))
    onset_recall = recall_score(labels[...,::3].reshape(-1), preds[...,::3].reshape(-1))
    offset_recall = recall_score(labels[...,1::3].reshape(-1), preds[...,1::3].reshape(-1))
    frame_recall = recall_score(labels[...,2::3].reshape(-1), preds[...,2::3].reshape(-1))

    return dict(
        onset_precision=onset_precision,
        offset_precision=offset_precision,
        frame_precision=frame_precision,
        onset_recall=onset_recall,
        offset_recall=offset_recall,
        frame_recall=frame_recall
    )

def train(model_file, train, eval, run, device):
    ckpt = torch.load(model_file)
    config = ckpt['config']
    model_state_dict = ckpt['model_state_dict']

    model = TranscriptionModel(config)
    model.load_state_dict(model_state_dict)
    model.to(device)

    model_size = model.combined_fc.in_features
    model.combined_fc = nn.Linear(model_size, OUTPUT_FEATURES)

    traind = SignalSampler(config, AudioDataset(config, "train", "labels/train"), len=8)
    evald = SignalSampler(config, AudioDataset(config, "test", "labels/train"), len=8)

    ta = transformers.TrainingArguments(output_dir="out", per_device_train_batch_size=8, per_device_eval_batch_size=8, num_train_epochs=100, report_to="wandb") # evaluation_strategy="epoch"
    trainer = transformers.Trainer(model, args=ta, train_dataset=traind, eval_dataset=evald, compute_metrics=compute_metrics)
    # trainer.add_callback(S3Callback())
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
