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
        self.cnt = 0

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

    def forward(self, x, labels): # batch x n_mels x length
        pitch_feature = self.pitch_feat_ext(x).transpose(1, 2).unsqueeze(1) # batch x chan x length x n_mels

        lang_batch = self.lang_model.run_on_batch({'audio': x})
        lang_feature = lang_batch['frame'].unsqueeze(1)

        x_lang = self.lang_conv_stack(lang_feature)
        x_lang_rnn = self.lang_rnn(x_lang)

        x_pitch = self.pitch_conv_stack(pitch_feature)
        x_pitch_rnn = self.pitch_rnn(x_pitch)

        print(x_pitch.shape, x_pitch_rnn.shape)

        x_combined = self.combined_rnn(torch.cat([x_pitch_rnn, x_lang_rnn], dim=2))
        x_combined = self.combined_fc(x_combined) # batch x n_frames x n_notes

        print(x_combined.shape, labels.shape)

        if labels is not None:
            x_combined = x_combined[:,:labels.shape[1],:]
            labels = labels.clamp(0.0, 1.0)
            plt.pcolor(labels[0].detach().cpu().numpy())
            plt.savefig(f"labels{self.cnt}.png")
            plt.pcolor(x_combined[0].detach().cpu().numpy())
            plt.savefig(f"preds{self.cnt}.png")
            self.cnt += 1
            loss = F.binary_cross_entropy_with_logits(x_combined.reshape(-1), labels.reshape(-1))
        else:
            loss = None
        
        print(loss)

        return loss, x_combined, labels
