import torch
from torch import nn
from .feature import FeatureExtractor
from .subnetworks import DilatedConvStack, BiLSTM
from .phonerec_model import PhonemeRecognitionModel
import torch.nn.functional as F
import matplotlib.pyplot as plt


class TranscriptionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pitch_input_features = config['n_mels']

        self.lang_model = PhonemeRecognitionModel(config['lang_model_config'])
        for p in self.lang_model.parameters():
            p.requires_grad = False

        self.lang_input_features = config['lang_model_config']['num_lbl'] - 1
        self.output_features = 3

        self.pitch_conv_stack, self.lang_conv_stack, self.pitch_rnn, self.lang_rnn, self.combined_rnn, self.combined_fc  = self._create_model(self.pitch_input_features, self.lang_input_features, self.output_features, config)

        self.pitch_feat_ext = FeatureExtractor(config)

        self.sr = config['sample_rate']
        self.win_length = config['win_length']
        self.hop_length = config['hop_length']
        self.pitch_sum = config['pitch_sum']
        # self.cnt = 0

    def _create_model(self, pitch_input_features, lang_input_features, output_features, config):
        model_complexity = config['model_complexity']
        model_size = model_complexity * 16

        pitch_conv_stack = DilatedConvStack(pitch_input_features, model_size)
        lang_conv_stack = DilatedConvStack(lang_input_features, model_size)

        pitch_rnn = BiLSTM(model_size, model_size // 2)
        lang_rnn = BiLSTM(model_size, model_size // 2)
        combined_rnn = BiLSTM(model_size * 2, model_size // 2) # batch x length x model_size
        combined_fc = nn.Linear(model_size, output_features) # batch x length x features

        return pitch_conv_stack, lang_conv_stack, pitch_rnn, lang_rnn, combined_rnn, combined_fc

    def forward(self, x, labels, notes): # batch x n_mels x length
        pitch_feature = self.pitch_feat_ext(x).transpose(1, 2).unsqueeze(1) # batch x chan x length x n_mels

        # print(pitch_feature.shape)
        # plt.pcolor(pitch_feature[0,0,:,:].detach().cpu().numpy())
        # plt.savefig(f"feats{self.cnt}.png")
        # plt.clf()


        lang_batch = self.lang_model.run_on_batch({'audio': x})
        lang_feature = lang_batch['frame'].unsqueeze(1)

        x_lang = self.lang_conv_stack(lang_feature)
        x_lang_rnn = self.lang_rnn(x_lang)

        x_pitch = self.pitch_conv_stack(pitch_feature)
        x_pitch_rnn = self.pitch_rnn(x_pitch)

        # print(x_pitch.shape, x_pitch_rnn.shape)

        x_combined = self.combined_rnn(torch.cat([x_pitch_rnn, x_lang_rnn], dim=2))
        x_combined = self.combined_fc(x_combined) # batch x n_frames x n_notes

        # print(x_combined.shape, labels.shape)
        # print(torch.sigmoid(x_combined).max())

        if labels is not None:
            if x_combined.shape != labels.shape:
                print("WARN: prediction shape doesn't match labels shape, expected:", labels.shape, "got", x_combined.shape)
                x_combined = x_combined[:,:labels.shape[1],:]
            labels = labels.clamp(0.0, 1.0)
            # plt.pcolor(labels[0].detach().cpu().numpy())
            # plt.savefig(f"labels{self.cnt}.png")
            # plt.clf()
            # plt.pcolor(x_combined[0].detach().cpu().numpy())
            # plt.savefig(f"preds{self.cnt}.png")
            # plt.clf()
            # self.cnt += 1
            onsets = x_combined[::3].transpose(1, 2)
            offsets = x_combined[1::3].transpose(1, 2)
            frames = x_combined[2::3].transpose(1, 2)

            onsets_lbl = labels[::3].transpose(1, 2)
            offsets_lbl = labels[1::3].transpose(1, 2)
            frames_lbl = labels[2::3].transpose(1, 2)

            w = torch.ones(onsets.shape[1]) * 3
            w[w.shape[0] - 1] = 1.0
            w = w.to('cuda:0')

            onsets_loss = F.cross_entropy(onsets, onsets_lbl, weight=w)
            offsets_loss = F.cross_entropy(offsets, offsets_lbl, weight=w)
            frames_loss = F.cross_entropy(frames, frames_lbl)

            loss = 2 * onsets_loss + offsets_loss + frames_loss
        else:
            loss = None
        
        # print(loss)

        return loss, x_combined, notes
