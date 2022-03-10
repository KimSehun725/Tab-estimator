import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet.nets.asr_interface import ASRInterface
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, mask_by_length
from ignite.utils import convert_tensor
import math
import random


def tab2pitch(tab):
    rel_string_pitches = [0, 5, 10, 15, 19, 24]
    pitch = torch.zeros(tab.shape[0], tab.shape[1], 44).to(tab.device)
    for string in range(6):
        pitch[:, :, rel_string_pitches[string]:rel_string_pitches[string]+20] += tab[:, :, string, :-1]
    pitch = torch.where(pitch > 1.0, 1.0, pitch.double())
    return pitch.float()


class calculate_score(nn.Module):
    def __init__(self):
        super(calculate_score, self).__init__()

    def scores(self, TP, TN, FP, FN):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = (2 * precision * recall) / (precision + recall)
        accuracy = (TP + TN) / (TP + FN + TN + FP)
        return torch.nan_to_num(precision), torch.nan_to_num(recall), torch.nan_to_num(F1), accuracy

    def forward(self, tab_pred, F0_pred, tab_gt, F0_gt, tab_gt_len):
        tab_pred = torch.squeeze(tab_pred, axis=0)
        F0_pred = torch.squeeze(F0_pred, axis=0)
        tab_gt = torch.squeeze(tab_gt, axis=0)
        F0_gt = torch.squeeze(F0_gt, axis=0)

        tab_onehot_index = torch.argmax(tab_pred, dim=2)
        F0_onehot_index = torch.argmax(F0_pred, dim=1)

        tab_pred = torch.zeros(tab_gt.shape).to(tab_gt.device)
        F0_pred = torch.zeros(F0_gt.shape).to(tab_gt.device)
        for time in range(tab_gt_len):
            F0_pred[time, F0_onehot_index[time]] = 1
            for string in range(6):
                tab_pred[time, string, tab_onehot_index[time, string]] = 1

        assert 1 >= tab_pred.all() >= 0
        assert 1 >= tab_gt.all() >= 0

        tab_TP = torch.sum((tab_pred[:, :, :-1] == 1)
                           & (tab_gt[:, :, :-1] == 1))
        tab_TN = torch.sum((tab_pred[:, :, :-1] == 0)
                           & (tab_gt[:, :, :-1] == 0))
        tab_FP = torch.sum((tab_pred[:, :, :-1] == 1)
                           & (tab_gt[:, :, :-1] == 0))
        tab_FN = torch.sum((tab_pred[:, :, :-1] == 0)
                           & (tab_gt[:, :, :-1] == 1))

        F0_TP = torch.sum((F0_pred[:, :-1] == 1) & (F0_gt[:, :-1] == 1))
        F0_TN = torch.sum((F0_pred[:, :-1] == 0) & (F0_gt[:, :-1] == 0))
        F0_FP = torch.sum((F0_pred[:, :-1] == 1) & (F0_gt[:, :-1] == 0))
        F0_FN = torch.sum((F0_pred[:, :-1] == 0) & (F0_gt[:, :-1] == 1))

        print_validation_scores = False
        if print_validation_scores == True:

            print(f"{F0_TP=}\t{tab_TP=}")
            print(f"{F0_TN=}\t{tab_TN=}")
            print(f"{F0_FP=}\t{tab_FP=}")
            print(f"{F0_FN=}\t{tab_FN=}")

        F0_prec, F0_recall, F0_F1, F0_acc = self.scores(
            F0_TP, F0_TN, F0_FP, F0_FN)
        tab_prec, tab_recall, tab_F1, tab_acc = self.scores(
            tab_TP, tab_TN, tab_FP, tab_FN)

        if print_validation_scores == True:
            print(f"{F0_prec=}\t{tab_prec=}")
            print(f"{F0_recall=}\t{tab_recall=}")
            print(f"{F0_F1=}\t{tab_F1=}")
            print(f"{F0_acc=}\t{tab_acc=}")
            print("========================")
        return


class GuidedAttentionLoss(nn.Module):
    """Guided attention loss function module.
    This module calculates the guided attention loss described
    in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_,
    which forces the attention to be diagonal.
    .. _`Efficiently Trainable Text-to-Speech System
        Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """

    def __init__(self, sigma=0.2, alpha=1.0, reset_always=True):
        """Initialize guided attention loss module.
        Args:
            sigma (float, optional): Standard deviation to control
                how close attention to a diagonal.
            alpha (float, optional): Scaling coefficient (lambda).
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.
        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lengths (B,).
            olens (LongTensor): Batch of output lengths (B,).
        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(
                att_ws.device
            )
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen, olen, self.sigma
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """Make guided attention mask.
        Examples:
            >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([5, 5])
            >>> guided_attn_mask
            tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                    [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                    [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                    [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                    [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
            >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([6, 3])
            >>> guided_attn_mask
            tensor([[0.0000, 0.2934, 0.7506],
                    [0.0831, 0.0831, 0.5422],
                    [0.2934, 0.0000, 0.2934],
                    [0.5422, 0.0831, 0.0831],
                    [0.7506, 0.2934, 0.0000],
                    [0.8858, 0.5422, 0.0831]])
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float().to(olen.device), grid_y.float().to(ilen.device)
        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma ** 2))
        )

    @staticmethod
    def _make_masks(ilens, olens):
        """Make masks indicating non-padded part.
        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor indicating non-padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        Examples:
            >>> ilens, olens = [5, 2], [8, 5]
            >>> _make_mask(ilens, olens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        in_masks = make_non_pad_mask(ilens)  # (B, T_in)
        out_masks = make_non_pad_mask(olens)  # (B, T_out)
        # (B, T_out, T_in)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)

    # (batch, time, 6, 21)


class CustomLoss(nn.Module):
    def __init__(self, mode, use_galoss):
        super(CustomLoss, self).__init__()
        weight = torch.ones(21) * 500
        weight[20] = 1
        self.cross_entropy = lambda gt, pred: -gt * \
            torch.log(pred + 1e-7) - (1 - gt) * torch.log(1 - pred + 1e-7)
        #self.CrossEntropy = nn.BCEWithLogitsLoss(weight=None, reduction='none')
        self.GALoss = GuidedAttentionLoss(sigma=0.4, alpha=1)
        self.mode = mode
        self.use_galoss = use_galoss

    def forward(self, frame_pred, frame_gt, note_pred, note_gt, attn, olens, note_len):
        batch_size = frame_gt.shape[0]
        attn_loss = 0
        frame_loss = self.cross_entropy(frame_gt, frame_pred)
        frame_loss = mask_by_length(frame_loss, olens)
        note_loss = self.cross_entropy(note_gt, note_pred)

        if self.mode == "F0":
            frame_loss = torch.sum(frame_loss) / torch.sum(olens) / 44
        elif self.mode == "tab":
            frame_loss = torch.sum(frame_loss) / torch.sum(olens) / 126

        note_loss = torch.mean(note_loss)

        for head in range(attn.shape[1]):
            attn_loss += self.GALoss(attn[:, head], olens, olens)

        print("loss : {:.4f}, {:.4f}, {:.4f}".format(
            frame_loss, note_loss, attn_loss))
        if self.use_galoss:
            loss = frame_loss + note_loss + attn_loss
        else:
            loss = frame_loss + note_loss

        return loss


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features, input_ch):
        super(ConvStack, self).__init__()

        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(input_ch, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features //
                      16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16,
                      output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) *
                      (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, X):
        y = self.cnn(X)
        y = y.transpose(1, 2).flatten(-2)
        y = self.fc(y)
        return y


class ESPNetTransformer(ASRInterface, torch.nn.Module):
    def __init__(self, mode, encoder_type, use_custom_decimation_func, use_conv_stack, n_bins, hop_length, sr, encoder_heads=1, encoder_layers=1, normalize_before=True):
        super(ESPNetTransformer, self).__init__()
        self.mode = mode
        self.use_custom_decimation_func = use_custom_decimation_func
        self.use_conv_stack = use_conv_stack
        self.hop_length = hop_length
        self.sr = sr
        self.encoder_output_size = 64
        self.n_encoder_ffn = 64
        self.encoder_attn_dropout = 0
        self.encoder_pos_dropout = 0.1
        self.conv_output_features = 16 * 32

        if use_conv_stack:
            self.convstack = ConvStack(n_bins, self.conv_output_features, 1)

        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(self.conv_output_features if use_conv_stack else n_bins,
                                              output_size=self.encoder_output_size,
                                              attention_heads=encoder_heads,
                                              linear_units=self.n_encoder_ffn,
                                              num_blocks=encoder_layers,
                                              positional_dropout_rate=self.encoder_pos_dropout,
                                              attention_dropout_rate=self.encoder_attn_dropout,
                                              input_layer='linear',
                                              positionwise_layer_type='conv1d',
                                              normalize_before=normalize_before)
        elif encoder_type == "conformer":
            self.encoder = ConformerEncoder(self.conv_output_features if use_conv_stack else n_bins,
                                            output_size=self.encoder_output_size,
                                            attention_heads=encoder_heads,
                                            linear_units=self.n_encoder_ffn,
                                            num_blocks=encoder_layers,
                                            attention_dropout_rate=self.encoder_attn_dropout,
                                            input_layer='linear',
                                            positionwise_layer_type='conv1d',
                                            positionwise_conv_kernel_size=3,
                                            normalize_before=normalize_before,
                                            macaron_style=False,
                                            rel_pos_type="latest",
                                            pos_enc_layer_type="rel_pos",
                                            selfattention_layer_type="rel_selfattn",
                                            cnn_module_kernel=3)

        if mode == "F0":
            self.frame_F0_output_layer = nn.Sequential(
                nn.Linear(self.encoder_output_size, 44),
                nn.Sigmoid()
            )
            self.note_F0_output_layer = nn.Sequential(
                nn.Linear(self.encoder_output_size, 44),
                nn.Sigmoid()
            )

        elif mode == "tab":
            self.frame_tab_output_layer = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(self.encoder_output_size, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 126)
            )

            self.note_tab_output_layer = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(self.encoder_output_size, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 126)
            )
            self.softmax_by_string = nn.Softmax(dim=3)

        self.note_encoder = ConformerEncoder(self.encoder_output_size,
                                             output_size=self.encoder_output_size,
                                             attention_heads=encoder_heads,
                                             linear_units=self.n_encoder_ffn,
                                             num_blocks=encoder_layers,
                                             attention_dropout_rate=self.encoder_attn_dropout,
                                             input_layer='linear',
                                             positionwise_layer_type='conv1d',
                                             positionwise_conv_kernel_size=3,
                                             normalize_before=normalize_before,
                                             macaron_style=False,
                                             rel_pos_type="latest",
                                             pos_enc_layer_type="rel_pos",
                                             selfattention_layer_type="rel_selfattn",
                                             cnn_module_kernel=3)

    def forward(self, src_pad, src_len, note_len, bpm):
        batch_size = src_pad.shape[0]

        if self.use_conv_stack:
            encoder_in = self.convstack(torch.unsqueeze(src_pad, dim=1))
        else:
            encoder_in = src_pad

        # Transformer or Conformer encoder
        memory, olens, _ = self.encoder(encoder_in, src_len)

        if self.use_custom_decimation_func:
            # custom decimation function
            with torch.no_grad():
                decimated_memory = self.notelevel_decimation(memory, bpm)
        else:
            # decimation using F.interpolate
            with torch.no_grad():
                # (batch, features, length)
                memory_cpy = torch.swapaxes(memory, 1, 2)
                decimated_memory = torch.zeros(
                    batch_size, self.encoder_output_size, 64).to(memory.device)
                for n_batch in range(batch_size):
                    decimated_memory[n_batch] = torch.squeeze(F.interpolate(
                        torch.unsqueeze(memory_cpy[n_batch, :, :olens[n_batch]], 0), size=64), 0)
                decimated_memory = torch.swapaxes(decimated_memory, 1, 2)

        if self.mode == "F0":
            # frame-level f0 output layer
            frame_F0_pred = self.frame_F0_output_layer(memory)
            # note-level f0 output layer
            decimated_memory, _, _ = self.note_encoder(
                decimated_memory, note_len)
            note_F0_pred = self.note_F0_output_layer(decimated_memory)
            return frame_F0_pred, note_F0_pred, olens

        elif self.mode == "tab":
            # frame-level tab output layer
            frame_tab_pred = self.frame_tab_output_layer(memory)
            frame_tab_pred = frame_tab_pred.view(batch_size, -1, 6, 21)
            frame_tab_pred = self.softmax_by_string(frame_tab_pred)

            # note-level tab output layer
            decimated_memory, _, _ = self.note_encoder(
                decimated_memory, note_len)
            note_tab_pred = self.note_tab_output_layer(decimated_memory)
            note_tab_pred = note_tab_pred.view(batch_size, -1, 6, 21)
            note_tab_pred = self.softmax_by_string(note_tab_pred)

            return frame_tab_pred, note_tab_pred, olens

        else:
            print("mode must be either 'F0' or 'tab'")

    def notelevel_decimation(self, memory, bpm):
        # memory(batch, len, features)
        padded_memory = F.pad(memory, (0, 0, 0, 10))  # for margin of error
        batch_size = memory.shape[0]
        feature_size = self.encoder_output_size
        output = torch.zeros(batch_size, 64, feature_size).to(memory.device)

        for n_batch in range(batch_size):
            frames_per_note = (
                (self.sr * 60) / (self.hop_length * 4 * bpm[n_batch])).float()
            for n_note in range(64):
                frame_start = n_note * frames_per_note
                start_floor = torch.floor(frame_start).int()
                start_ceil = torch.ceil(frame_start).int()
                frame_end = (n_note + 1) * frames_per_note
                end_floor = torch.floor(frame_end).int()
                end_ceil = torch.ceil(frame_end).int()

                sum_prob = padded_memory[n_batch,
                                         start_floor, :] * (start_ceil - frame_start)
                sum_prob = torch.add(sum_prob, torch.sum(
                    padded_memory[n_batch, start_ceil:end_floor, :], dim=0))
                sum_prob = torch.add(
                    sum_prob, padded_memory[n_batch, end_floor, :] * (frame_end - end_floor))
                mean_prob = sum_prob / frames_per_note
                output[n_batch, n_note] = mean_prob

        return output
