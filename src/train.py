import datetime
import torch
import numpy as np
import yaml
import glob
import os
import shutil
import math
import random
import time
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from ignite.engine import Engine, Events
from ignite.metrics import Loss, Metric
from ignite.utils import convert_tensor
import tensorboardX
from torch_lr_finder import LRFinder
from random import choices, sample, seed, shuffle
from network import TabEstimator, CustomLoss, calculate_score
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class LossWrapper(Loss):
    """
    Wrapper for ignite.metrics.Loss
    """

    def __init__(self, loss_fn, output_transform=lambda x: x,
                 batch_size=lambda x: len(x)):
        super(LossWrapper, self).__init__(
            loss_fn, output_transform=output_transform, batch_size=batch_size)

    def update(self, output):
        frame_pred, frame_gt, note_pred, note_gt, attn_map, olens, note_len = output
        loss = self._loss_fn(frame_pred, frame_gt.float(
        ), note_pred, note_gt.float(), attn_map, olens, note_len)
        average_loss = loss

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return an average loss.")

        N = self._batch_size(frame_gt)
        self._sum += average_loss.item() * N
        self._num_examples += N


class CustomDataset(Dataset):
    def __init__(self, data_list, mode, input_feature_type):
        self.data_list = data_list
        self.mode = mode
        self.input_feature_type = input_feature_type

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = np.load(self.data_list[index])
        if self.input_feature_type == "cqt":
            input_features = data["cqt"]
        elif self.input_feature_type == "melspec":
            input_features = data["mel_spec"]

        if self.mode == "F0":
            note_gt = data["F0"]
            frame_gt = data["frame_F0"]

        elif self.mode == "tab":
            note_gt = data["tab"]
            frame_gt = data["frame_tab"]

        bpm = data["tempo"]

        frame_len = input_features.shape[0]
        note_len = note_gt.shape[0]

        return input_features, frame_gt, note_gt, frame_len, note_len, bpm


def _prepare_batch(batch, mode, device=None, non_blocking=False):
    """
    Prepare batch for training: pass to a device with options.
    """
    input_features, frame_gt, note_gt, frame_len, note_len, bpm = batch

    return (convert_tensor(input_features, device=device, non_blocking=non_blocking),
            convert_tensor(frame_gt, device=device, non_blocking=non_blocking),
            convert_tensor(note_gt, device=device, non_blocking=non_blocking),
            convert_tensor(frame_len, device=device,
                           non_blocking=non_blocking),
            convert_tensor(note_len, device=device, non_blocking=non_blocking),
            convert_tensor(bpm, device=device, non_blocking=non_blocking))


def F0_pad_collate(batch):
    input_features, frame_F0, note_F0, frame_len, note_len, bpm = zip(*batch)
    batch_size = len(input_features)
    frame_maxlen, note_maxlen = 0, 0
    for sample_framelen, sample_notelen in zip(frame_len, note_len):
        if sample_framelen > frame_maxlen:
            frame_maxlen = sample_framelen
        if sample_notelen > note_maxlen:
            note_maxlen = sample_notelen
    frame_len = np.asarray(frame_len)
    note_len = np.asarray(note_len)
    bpm = np.asarray(bpm)
    for batch_n in range(batch_size):
        frame_padlen = frame_maxlen - frame_len[batch_n]
        note_padlen = note_maxlen - note_len[batch_n]
        padded_input_features = np.pad(
            input_features[batch_n], [(0, frame_padlen), (0, 0)], 'constant')
        padded_note_F0 = np.pad(
            note_F0[batch_n], [(0, note_padlen), (0, 0)], 'constant')
        padded_frame_F0 = np.pad(
            frame_F0[batch_n], [(0, frame_padlen), (0, 0)], 'constant')

        if batch_n == 0:
            padded_input_features_out = np.expand_dims(
                padded_input_features, axis=0)
            padded_note_F0_out = np.expand_dims(padded_note_F0, axis=0)
            padded_frame_F0_out = np.expand_dims(padded_frame_F0, axis=0)
        else:
            padded_input_features_out = np.append(
                padded_input_features_out, np.expand_dims(padded_input_features, axis=0), axis=0)
            padded_note_F0_out = np.append(
                padded_note_F0_out, np.expand_dims(padded_note_F0, axis=0), axis=0)
            padded_frame_F0_out = np.append(
                padded_frame_F0_out, np.expand_dims(padded_frame_F0, axis=0), axis=0)

    # reverse sort by length
    sort_idx = np.argsort(frame_len)[::-1]
    padded_input_features_out = np.take(
        padded_input_features_out, sort_idx, axis=0)
    padded_note_F0_out = np.take(padded_note_F0_out, sort_idx, axis=0)
    padded_frame_F0_out = np.take(padded_frame_F0_out, sort_idx, axis=0)
    frame_len = np.take(frame_len, sort_idx, axis=0)
    note_len = np.take(note_len, sort_idx, axis=0)
    bpm = np.take(bpm, sort_idx, axis=0)

    return torch.from_numpy(padded_input_features_out), torch.from_numpy(padded_frame_F0_out), torch.from_numpy(padded_note_F0_out), torch.from_numpy(frame_len), torch.from_numpy(note_len), torch.from_numpy(bpm)


def tab_pad_collate(batch):
    input_features, frame_tab, note_tab, frame_len, note_len, bpm = zip(*batch)
    batch_size = len(input_features)

    frame_maxlen, note_maxlen = np.max(frame_len), np.max(note_len)

    frame_len = np.asarray(frame_len)
    note_len = np.asarray(note_len)
    bpm = np.asarray(bpm)
    for batch_n in range(batch_size):
        frame_padlen = frame_maxlen - frame_len[batch_n]
        note_padlen = note_maxlen - note_len[batch_n]
        padded_input_features = np.pad(
            input_features[batch_n], [(0, frame_padlen), (0, 0)], 'constant')
        padded_note_tab = np.pad(
            note_tab[batch_n], [(0, note_padlen), (0, 0), (0, 0)], 'constant')
        padded_frame_tab = np.pad(
            frame_tab[batch_n], [(0, frame_padlen), (0, 0), (0, 0)], 'constant')

        if batch_n == 0:
            padded_input_features_out = np.expand_dims(
                padded_input_features, axis=0)
            padded_note_tab_out = np.expand_dims(padded_note_tab, axis=0)
            padded_frame_tab_out = np.expand_dims(padded_frame_tab, axis=0)

        else:
            padded_input_features_out = np.append(
                padded_input_features_out, np.expand_dims(padded_input_features, axis=0), axis=0)
            padded_note_tab_out = np.append(
                padded_note_tab_out, np.expand_dims(padded_note_tab, axis=0), axis=0)
            padded_frame_tab_out = np.append(
                padded_frame_tab_out, np.expand_dims(padded_frame_tab, axis=0), axis=0)

    # reverse sort by length
    sort_idx = np.argsort(frame_len)[::-1]
    padded_input_features_out = np.take(
        padded_input_features_out, sort_idx, axis=0)
    padded_note_tab_out = np.take(padded_note_tab_out, sort_idx, axis=0)
    padded_frame_tab_out = np.take(padded_frame_tab_out, sort_idx, axis=0)
    frame_len = np.take(frame_len, sort_idx, axis=0)
    note_len = np.take(note_len, sort_idx, axis=0)
    bpm = np.take(bpm, sort_idx, axis=0)

    return torch.from_numpy(padded_input_features_out), torch.from_numpy(padded_frame_tab_out), torch.from_numpy(padded_note_tab_out), torch.from_numpy(frame_len), torch.from_numpy(note_len), torch.from_numpy(bpm)


def train(mode, input_feature_type, encoder_type, use_custom_decimation_func, use_conv_stack, use_galoss, test_num, train_data_list, valid_data_list, tensorboard_dir, model_dir, epoch, lr,  d_model, encoder_heads, encoder_layers, n_cores, device, n_bins, hop_length, sr):
    writer = tensorboardX.SummaryWriter(tensorboard_dir)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    model = TabEstimator(mode, encoder_type, use_custom_decimation_func, use_conv_stack, n_bins, hop_length, sr, encoder_heads=encoder_heads,
                              encoder_layers=encoder_layers)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    criterion = CustomLoss(mode, use_galoss)
    metrics_fn = calculate_score()

    optimizer = optim.RAdam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 32, gamma=0.5, verbose=False)

    model.cuda()
    criterion.cuda()
    metrics_fn.cuda()
    device = "cuda"

    train_dataset = CustomDataset(train_data_list, mode, input_feature_type)
    valid_dataset = CustomDataset(valid_data_list, mode, input_feature_type)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=(F0_pad_collate if mode == "F0" else tab_pad_collate),
        num_workers=n_cores,
        pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=n_cores,
        pin_memory=True)

    class Loss_container():
        def __init__(self):
            self.loss_value = 0
            self.epoch_loss_value = 0

        def _reset_itr(self):
            self.loss_value = 0

        def update_itr(self):
            self.epoch_loss_value += self.loss_value
            self._reset_itr()

        def reset_epoch(self):
            self.epoch_loss_value = 0

    loss_container = Loss_container()

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        if mode == "F0":
            padded_input_features, padded_frame_F0_gt, note_F0_gt, frame_len, note_len, bpm = _prepare_batch(
                batch, mode, device=device)
        elif mode == "tab":
            padded_input_features, padded_frame_tab_gt, note_tab_gt, frame_len, note_len, bpm = _prepare_batch(
                batch, mode, device=device)

        frame_pred, note_pred, olens = model(
            padded_input_features.float(), frame_len, note_len, bpm)
        encoder_self_attn_map = model.encoder.encoders._modules['0']._modules['self_attn'].attn
        if mode == "F0":
            loss = criterion(frame_pred, padded_frame_F0_gt, note_pred,
                             note_F0_gt, encoder_self_attn_map, olens, note_len)
        elif mode == "tab":
            loss = criterion(frame_pred, padded_frame_tab_gt, note_pred,
                             note_tab_gt, encoder_self_attn_map, olens, note_len)
        loss_container.loss_value += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        return loss.item()

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            if mode == "F0":
                padded_input_features, padded_frame_F0_gt, note_F0_gt, frame_len, note_len, bpm = _prepare_batch(
                    batch, mode, device=device)
            elif mode == "tab":
                padded_input_features, padded_frame_tab_gt, note_tab_gt, frame_len, note_len, bpm = _prepare_batch(
                    batch, mode, device=device)

            frame_pred, note_pred, olens = model(
                padded_input_features.float(), frame_len, note_len, bpm)
            encoder_self_attn_map = model.encoder.encoders._modules['0']._modules['self_attn'].attn
            if mode == "F0":
                return frame_pred, padded_frame_F0_gt, note_pred, note_F0_gt, encoder_self_attn_map, olens, note_len
            elif mode == "tab":
                return frame_pred, padded_frame_tab_gt, note_pred, note_tab_gt, encoder_self_attn_map, olens, note_len

    trainer = Engine(_update)
    evaluator = Engine(_inference)
    metrics = {"Loss": LossWrapper(criterion)}
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        writer.add_scalar("train/loss", trainer.state.output,
                          trainer.state.iteration)
        loss_container.update_itr()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        avg_loss = loss_container.epoch_loss_value / len(train_loader)
        print("Training Results - Epoch: {}  Avg loss: {:.4f}"
              .format(trainer.state.epoch, avg_loss))
        writer.add_scalar("train/avg_loss",
                          avg_loss, trainer.state.epoch)
        loss_container.reset_epoch()
        if trainer.state.epoch % 32 == 0:
            modelname = "epoch{}.model".format(trainer.state.epoch)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), os.path.join(model_dir, modelname))
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg loss: {:.4f}"
              .format(trainer.state.epoch, metrics["Loss"]))
        writer.add_scalar("valid/avg_loss",
                          metrics["Loss"], trainer.state.epoch)

    trainer.run(train_loader, max_epochs=epoch)
    writer.close()
    return


def main(mode, input_feature_type, encoder_type, use_custom_decimation_func, use_conv_stack, use_galoss, train_ratio, note_resolution, epoch, lr, seed_, d_model, encoder_heads, encoder_layers, n_cores, cqt_n_bins, hop_length, sr):
    data_path = os.path.join(
        "data", "npz", f"original", "split", "*.npz")
    data_list = np.array(glob.glob(data_path, recursive=True))

    now = datetime.datetime.now()
    tensorboard_dir = os.path.join("tensorboard", "{0:%Y%m%d%H%M}".format(now))
    model_dir = os.path.join("model", "{0:%Y%m%d%H%M}".format(now))
    os.makedirs(model_dir)

    if input_feature_type == "cqt":
        n_bins = cqt_n_bins
    elif input_feature_type == "melspec":
        n_bins = 128

    shutil.copyfile("src/config.yaml", model_dir + "/config.yaml")

    if torch.cuda.is_available():
        device = 'cuda'
        for test_num in range(6):
            dev_data_list = [datapath for datapath in data_list if not(
                os.path.split(datapath)[1].startswith(f"0{test_num}_"))]
            random.shuffle(dev_data_list)
            train_data_list = dev_data_list[:int(
                round(len(dev_data_list) * train_ratio))]
            valid_data_list = dev_data_list[int(
                round(len(dev_data_list) * train_ratio)):]
            tensorboard_dir = os.path.join(
                "tensorboard", "{0:%Y%m%d%H%M}".format(now), f"testNo0{test_num}")
            model_dir = os.path.join(
                "model", "{0:%Y%m%d%H%M}".format(now), f"testNo0{test_num}")
            train(mode, input_feature_type, encoder_type, use_custom_decimation_func, use_conv_stack, use_galoss, test_num, train_data_list, valid_data_list, tensorboard_dir, model_dir,
                  epoch, lr, d_model, encoder_heads, encoder_layers, n_cores, device, n_bins, hop_length, sr)
    else:
        raise EnvironmentError("CUDA is not avaible")

    return


if __name__ == "__main__":
    with open("src/config.yaml") as f:
        obj = yaml.safe_load(f)
        hop_length = obj["hop_length"]
        sr = obj["down_sampling_rate"]
        train_ratio = obj["train_ratio"]
        note_resolution = obj["note_resolution"]
        cqt_n_bins = obj["cqt_n_bins"]
        epoch = obj["epoch"]
        lr = obj["lr"]
        seed_ = obj["seed_"]
        d_model = obj["d_model"]
        encoder_heads = obj["encoder_heads"]
        encoder_layers = obj["encoder_layers"]
        n_cores = obj["n_cores"]
        input_feature_type = obj["input_feature_type"]
        mode = obj["mode"]
        encoder_type = obj["encoder_type"]
        use_custom_decimation_func = obj["use_custom_decimation_func"]
        use_conv_stack = obj["use_conv_stack"]
        use_galoss = obj["use_galoss"]

    assert input_feature_type == "cqt" or input_feature_type == "melspec"
    assert mode == "F0" or mode == "tab"
    assert encoder_type == "transformer" or encoder_type == "conformer"

    main(mode, input_feature_type, encoder_type, use_custom_decimation_func, use_conv_stack, use_galoss, train_ratio, note_resolution, epoch, lr, seed_, d_model, encoder_heads,
         encoder_layers, n_cores, cqt_n_bins, hop_length, sr)
