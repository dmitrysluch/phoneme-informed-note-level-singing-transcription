from torch import nn
import numpy as np
import torch
import wquantiles
import librosa
import scipy.special as sc


SENSITIVITY = 2
MIN_NOTE_LEN_SEC = 0.05

class FramewiseDecoder:
    def __init__(self, config):
        self.sr = config['sample_rate']

        self.win_length = config['win_length']
        self.hop_length = config['hop_length']

        self.onset_threshold = config['onset_threshold']
        self.offset_threshold = config['offset_threshold']

        self.pitch_sum = config['pitch_sum']

        self.activation = nn.Sigmoid()

    def decode(self, pred):
        pred = np.copy(pred)
        pred[:,-3:] /= SENSITIVITY
        onsets = sc.softmax(pred[:,::3], axis=-1)
        offsets = sc.softmax(pred[:,1::3], axis=-1)
        frames = sc.softmax(pred[:,2::3], axis=-1)

        NUM_PITCHES = onsets.shape[1]

        onset_i = np.argmax(onsets, axis=-1)
        offset_i = np.argmax(offsets, axis=-1)
        frames_i = np.argmax(frames, axis=-1)

        onset_peaks = self._get_peaks(NUM_PITCHES, onsets, onset_i)
        onset_peak_mask = np.zeros(onset_i.shape, dtype=bool)
        onset_peak_mask[onset_peaks] = 1

        offset_peaks = self._get_peaks(NUM_PITCHES, offsets, offset_i)
        offset_peak_mask = np.zeros(offset_i.shape, dtype=bool)
        offset_peak_mask[offset_peaks] = 1
        offset_peak_mask |= onset_peak_mask

        intervals = []
        pitches = []
        min_note_len_frames = int(MIN_NOTE_LEN_SEC * self.sr / self.hop_length)

        for peak in onset_peaks:
            pitch = onset_i[peak]
            offset = peak + 1
            while offset < len(frames_i) and frames_i[offset] == pitch and not offset_peak_mask[offset]:
            # while offset < len(frames_i) and not offset_peak_mask[offset]:
                offset += 1
            if offset - peak < min_note_len_frames:
                continue
            intervals.append((peak, offset))
            pitches.append(pitch)

        intervals = np.array(intervals).astype('float64').reshape(-1, 2)
        pitches = np.array(pitches)
        intervals *= self.hop_length / self.sr
        return intervals, pitches

    def _get_peaks(self, NUM_PITCHES, prob, arr):
        max_i = -1
        peaks = []
        for i, pitch in enumerate(arr):
            if pitch != NUM_PITCHES - 1:
                if max_i == -1 or prob[i, pitch] > prob[max_i, pitch]:
                    max_i = i
            elif max_i != -1:
                peaks.append(max_i)
                max_i = -1
        if max_i != -1:
            peaks.append(max_i)
        return peaks
