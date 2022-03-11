import glob
import numpy as np
import os
import torch
import pandas as pd
import tqdm
import yaml
from network import TabEstimator
from visualize import visualize
import tqdm
from matplotlib import lines as mlines, pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
import seaborn as sns
import librosa
import argparse
from sklearn.metrics import precision_recall_fscore_support


def calculate_metrics(pred, gt):
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt, pred, average='binary')

    return precision, recall, f1


def tab2pitch(tab):
    rel_string_pitches = [0, 5, 10, 15, 19, 24]
    argmax_index = np.argmax(tab, axis=2)
    pitch = np.zeros((len(tab), 44))
    for time in range(len(tab)):
        for string in range(6):
            if argmax_index[time, string] < 20:
                pitch[time, argmax_index[time, string] +
                      rel_string_pitches[string]] = 1

    return pitch


def TDR(tab_pred, tab_gt, F0_gt):
    F0_from_tab_pred = tab2pitch(tab_pred)

    TP_tab = np.multiply(tab_gt[:, :, :-1], tab_pred[:, :, :-1]).sum()
    TP_F0 = np.multiply(F0_gt, F0_from_tab_pred).sum()
    tdr = TP_tab / TP_F0
    return tdr


def calc_score(test_num, trained_model, use_model_epoch, config_path, plot_results=False, input_as_random_noize=False,  make_notelvl_from_framelvl=False, verbose=True):
    with open(config_path) as f:
        obj = yaml.safe_load(f)
        note_resolution = obj["note_resolution"]
        down_sampling_rate = obj["down_sampling_rate"]
        bins_per_octave = obj["bins_per_octave"]
        hop_length = obj["hop_length"]
        cqt_n_bins = obj["cqt_n_bins"]
        d_model = obj["d_model"]
        encoder_heads = obj["encoder_heads"]
        encoder_layers = obj["encoder_layers"]
        input_feature_type = obj["input_feature_type"]
        mode = obj["mode"]
        encoder_type = obj["encoder_type"]
        use_custom_decimation_func = obj["use_custom_decimation_func"]
        use_conv_stack = obj["use_conv_stack"]

    kwargs = {
        "note_resolution": note_resolution,
        "down_sampling_rate": down_sampling_rate,
        "hop_length": hop_length,
        "bins_per_octave": bins_per_octave
    }
    model_path = f"model/{trained_model}/testNo0{test_num}/epoch{use_model_epoch}.model"
    existing_dev_files = glob.glob(f"visualize/dev/attn_map/0{test_num}/*")
    for f in existing_dev_files:
        os.remove(f)

    if input_feature_type == "cqt":
        n_bins = cqt_n_bins
    elif input_feature_type == "melspec":
        n_bins = 128

    model = TabEstimator(mode, encoder_type, use_custom_decimation_func, use_conv_stack, n_bins, hop_length, down_sampling_rate, encoder_heads=encoder_heads,
                              encoder_layers=encoder_layers)
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device("cpu")))
    model.eval()
    if verbose:
        print(f"{test_num=}, {mode=}")
    frame_sum_precision, frame_sum_recall, frame_sum_f1 = 0, 0, 0
    note_sum_precision, note_sum_recall, note_sum_f1 = 0, 0, 0
    if mode == "tab":
        frame_sum_F0_from_tab_precision, frame_sum_F0_from_tab_recall, frame_sum_F0_from_tab_f1 = 0, 0, 0
        note_sum_F0_from_tab_precision, note_sum_F0_from_tab_recall, note_sum_F0_from_tab_f1 = 0, 0, 0
        frame_sum_tdr, note_sum_tdr = 0, 0
    test_data_path = os.path.join(
        "data", "npz", f"original", "split", f"0{test_num}_*.npz")
    test_data_list = np.array(glob.glob(test_data_path, recursive=True))
    # every test file for loop start
    frame_concat_pred = np.array([])
    frame_concat_gt = np.array([])

    for npz_filename in tqdm.tqdm(test_data_list):
        npz_file = np.load(npz_filename)
        len_in_notes = npz_file["len_in_notes"]
        if input_feature_type == "cqt":
            input_features = torch.from_numpy(npz_file["cqt"])
        elif input_feature_type == "melspec":
            input_features = torch.from_numpy(npz_file["mel_spec"])
        input_features = torch.unsqueeze(input_features, 0)

        if mode == "F0":
            note_gt = torch.from_numpy(npz_file["F0"])
            frame_gt = torch.from_numpy(npz_file["frame_F0"])

        elif mode == "tab":
            note_gt = torch.from_numpy(npz_file["tab"])
            frame_gt = torch.from_numpy(npz_file["frame_tab"])

            note_F0_gt = npz_file["F0"]
            frame_F0_gt = npz_file["frame_F0"]
        else:
            print("mode must be either 'F0' or 'tab'")
            return

        bpm = torch.from_numpy(npz_file["tempo"])
        bpm = torch.unsqueeze(bpm, 0)

        frame_len = np.arange(1)
        note_len = np.arange(1)
        frame_len[0] = input_features.shape[1]
        note_len[0] = note_gt.shape[0]

        frame_len = torch.zeros(1)
        frame_len[0] = input_features.shape[1]

        if input_as_random_noize == True:
            input_features = torch.rand(input_features.shape)

        # prediction
        with torch.no_grad():
            frame_pred, note_pred, olens = model(
                input_features.float(), frame_len, note_len, bpm)

        if make_notelvl_from_framelvl:
            if mode == "F0":
                note_pred = model.notelevel_interpolation(frame_pred, bpm)
            elif mode == "tab":
                note_pred = model.notelevel_interpolation(
                    torch.flatten(frame_pred, start_dim=2), bpm)
                note_pred = torch.reshape(1, -1, 6, 21)

        # (batch, len, ...) -> (len, ...) & probability -> one-hot
        input_features = torch.squeeze(
            input_features, 0).detach().numpy().copy()
        if mode == "F0":
            frame_pred = torch.squeeze(frame_pred, 0)
            note_pred = torch.squeeze(note_pred, 0)

            frame_gt = frame_gt.detach().numpy()
            note_gt = note_gt.detach().numpy()

            frame_pred = np.where(frame_pred.detach().numpy() > 0.5, 1, 0)
            note_pred = np.where(note_pred.detach().numpy() > 0.5, 1, 0)

        elif mode == "tab":

            frame_pred = torch.squeeze(frame_pred, 0)
            argmax_index = np.argmax(frame_pred.detach().numpy(), axis=2)
            frame_pred = np.zeros((len(frame_pred), 6, 21))
            for frame in range(len(frame_pred)):
                for string in range(6):
                    frame_pred[frame, string, argmax_index[frame, string]] = 1
            frame_F0_from_tab_pred = tab2pitch(frame_pred)

            note_pred = torch.squeeze(note_pred, 0)
            argmax_index = np.argmax(note_pred.detach().numpy(), axis=2)
            note_pred = np.zeros((len(note_pred), 6, 21))
            for note in range(len(note_pred)):
                for string in range(6):
                    note_pred[note, string, argmax_index[note, string]] = 1
            note_F0_from_tab_pred = tab2pitch(note_pred)

            frame_gt = frame_gt.detach().numpy()
            note_gt = note_gt.detach().numpy()

            frame_tdr = TDR(frame_pred, frame_gt, frame_F0_gt)
            note_tdr = TDR(note_pred, note_gt, note_F0_gt)

            frame_sum_tdr += frame_tdr
            note_sum_tdr += note_tdr

        # getting attention map
        attn_map = model.encoder.encoders._modules['0']._modules['self_attn'].attn
        for n_layer in range(1, encoder_layers):
            attn_map = torch.cat(
                (attn_map, model.encoder.encoders._modules[f'{n_layer}']._modules['self_attn'].attn), dim=0)

        # save results as npz file
        if mode == "F0":
            npz_save_dir = os.path.join(
                "result", "F0", f"{trained_model}_epoch{use_model_epoch}", "npz", f"test_0{test_num}")
            npz_save_filename = os.path.join(
                npz_save_dir, os.path.split(npz_filename)[1])
            if not(os.path.exists(npz_save_dir)):
                os.makedirs(npz_save_dir)
            np.savez_compressed(npz_save_filename,
                                input_features=input_features,
                                frame_F0_pred=frame_pred,
                                frame_F0_gt=frame_gt,
                                note_F0_pred=note_pred,
                                note_F0_gt=note_gt,
                                attn_map=attn_map)
        elif mode == "tab":
            npz_save_dir = os.path.join(
                "result", "tab", f"{trained_model}_epoch{use_model_epoch}", "npz", f"test_0{test_num}")
            npz_save_filename = os.path.join(
                npz_save_dir, os.path.split(npz_filename)[1])
            if not(os.path.exists(npz_save_dir)):
                os.makedirs(npz_save_dir)
            np.savez_compressed(npz_save_filename,
                                input_features=input_features,
                                frame_tab_pred=frame_pred,
                                frame_tab_gt=frame_gt,
                                note_tab_pred=note_pred,
                                note_tab_gt=note_gt,
                                frame_F0_from_tab_pred=frame_F0_from_tab_pred,
                                frame_F0_gt=frame_F0_gt,
                                note_F0_from_tab_pred=note_F0_from_tab_pred,
                                note_F0_gt=note_F0_gt,
                                attn_map=attn_map)

        # flatten and calculate metrics
        if mode == "F0":
            frame_pred = frame_pred.flatten()
            frame_gt = frame_gt.flatten()
            note_pred = note_pred.flatten()
            note_gt = note_gt.flatten()
        if mode == "tab":
            # remove 'not played' class
            frame_pred = frame_pred[:, :, :-1].flatten()
            frame_gt = frame_gt[:, :, :-1].flatten()
            note_pred = note_pred[:, :, :-1].flatten()
            note_gt = note_gt[:, :, :-1].flatten()

            frame_F0_from_tab_pred = frame_F0_from_tab_pred.flatten()
            frame_F0_gt = frame_F0_gt.flatten()
            note_F0_from_tab_pred = note_F0_from_tab_pred.flatten()
            note_F0_gt = note_F0_gt.flatten()

        frame_concat_pred = np.concatenate(
            (frame_concat_pred, frame_pred), axis=None)
        frame_concat_gt = np.concatenate(
            (frame_concat_gt, frame_gt), axis=None)

        frame_precision, frame_recall, frame_f1 = calculate_metrics(
            frame_pred, frame_gt)
        note_precision, note_recall, note_f1 = calculate_metrics(
            note_pred, note_gt)
        if mode == "tab":
            frame_F0_from_tab_precision, frame_F0_from_tab_recall, frame_F0_from_tab_f1 = calculate_metrics(
                frame_F0_from_tab_pred, frame_F0_gt)
            note_F0_from_tab_precision, note_F0_from_tab_recall, note_F0_from_tab_f1 = calculate_metrics(
                note_F0_from_tab_pred, note_F0_gt)

        frame_sum_precision += frame_precision
        frame_sum_recall += frame_recall
        frame_sum_f1 += frame_f1

        note_sum_precision += note_precision
        note_sum_recall += note_recall
        note_sum_f1 += note_f1
        if mode == "tab":
            frame_sum_F0_from_tab_precision += frame_F0_from_tab_precision
            frame_sum_F0_from_tab_recall += frame_F0_from_tab_recall
            frame_sum_F0_from_tab_f1 += frame_F0_from_tab_f1

            note_sum_F0_from_tab_precision += note_F0_from_tab_precision
            note_sum_F0_from_tab_recall += note_F0_from_tab_recall
            note_sum_F0_from_tab_f1 += note_F0_from_tab_f1

    frame_avg_precision = frame_sum_precision / len(test_data_list)
    frame_avg_recall = frame_sum_recall / len(test_data_list)
    frame_avg_f1 = frame_sum_f1 / len(test_data_list)

    note_avg_precision = note_sum_precision / len(test_data_list)
    note_avg_recall = note_sum_recall / len(test_data_list)
    note_avg_f1 = note_sum_f1 / len(test_data_list)

    if verbose:
        print("frame_avg_p = {:.4f}, frame_avg_r = {:.4f}, frame_avg_f1 = {:.4f}".format(
            frame_avg_precision, frame_avg_recall, frame_avg_f1))
        print("note_avg_p = {:.4f}, note_avg_r = {:.4f}, note_avg_f1 = {:.4f}".format(
            note_avg_precision, note_avg_recall, note_avg_f1))

    if mode == "tab":
        frame_avg_F0_from_tab_precision = frame_sum_F0_from_tab_precision / \
            len(test_data_list)
        frame_avg_F0_from_tab_recall = frame_sum_F0_from_tab_recall / \
            len(test_data_list)
        frame_avg_F0_from_tab_f1 = frame_sum_F0_from_tab_f1 / \
            len(test_data_list)

        note_avg_F0_from_tab_precision = note_sum_F0_from_tab_precision / \
            len(test_data_list)
        note_avg_F0_from_tab_recall = note_sum_F0_from_tab_recall / \
            len(test_data_list)
        note_avg_F0_from_tab_f1 = note_sum_F0_from_tab_f1 / len(test_data_list)

        frame_avg_tdr = frame_sum_tdr / len(test_data_list)
        note_avg_tdr = note_sum_tdr / len(test_data_list)

    frame_concat_precision, frame_concat_recall, frame_concat_f1 = calculate_metrics(
        frame_concat_pred, frame_concat_gt)

    if mode == "F0":
        result = pd.DataFrame([[frame_avg_precision, frame_avg_recall, frame_avg_f1,
                                frame_concat_precision, frame_concat_recall, frame_concat_f1,
                                note_avg_precision, note_avg_recall, note_avg_f1]],
                              columns=["frame_segment_avg_F0_p", "frame_segment_avg_F0_r", "frame_segment_avg_F0_f",
                                       "frame_frame_avg_F0_p", "frame_frame_avg_F0_r", "frame_frame_avg_F0_f",
                                       "note_avg_F0_p", "note_avg_F0_r", "note_avg_F0_f"],
                              index=[f"No0{test_num}"])
    elif mode == "tab":
        result = pd.DataFrame([[frame_avg_precision, frame_avg_recall, frame_avg_f1,
                                frame_concat_precision, frame_concat_recall, frame_concat_f1,
                                note_avg_precision, note_avg_recall, note_avg_f1,
                                frame_avg_F0_from_tab_precision, frame_avg_F0_from_tab_recall, frame_avg_F0_from_tab_f1,
                                note_avg_F0_from_tab_precision, note_avg_F0_from_tab_recall, note_avg_F0_from_tab_f1,
                                frame_avg_tdr, note_avg_tdr]],
                              columns=["frame_segment_avg_tab_p", "frame_segment_avg_tab_r", "frame_segment_avg_tab_f",
                                       "frame_frame_avg_tab_p", "frame_frame_avg_tab_r", "frame_frame_avg_tab_f",
                                       "note_avg_tab_p", "note_avg_tab_r", "note_avg_tab_f",
                                       "frame_avg_F0_from_tab_p", "frame_avg_F0_from_tab_r", "frame_avg_F0_from_tab_f",
                                       "note_avg_F0_from_tab_p", "note_avg_F0_from_tab_r", "note_avg_F0_from_tab_f",
                                       "frame_avg_tdr", "note_avg_tdr"],
                              index=[f"No0{test_num}"])
    return result


def main():
    parser = argparse.ArgumentParser(
        description='code for predicting and saving results')
    parser.add_argument("model", type=str,
                        help="name of trained model: ex) 202201010000")
    parser.add_argument("epoch", type=int,
                        help="number of model epoch to use: ex) 64")
    parser.add_argument("-v", "--verbose", help="option for verbosity: -v to turn on verbosity",
                        action="store_true", required=False, default=False)
    args = parser.parse_args()

    trained_model = args.model
    use_model_epoch = args.epoch
    verbose = args.verbose

    result_path = os.path.join("result")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    input_as_random_noize = False
    plot_results = False
    make_notelvl_from_framelvl = False

    result = pd.DataFrame()
    config_path = os.path.join("model", f"{trained_model}", "config.yaml")

    with open(config_path) as f:
        obj = yaml.safe_load(f)
        mode = obj["mode"]

    csv_path = os.path.join(result_path, f"{mode}", trained_model +
                            f"_epoch{use_model_epoch}", "metrics.csv")
    for test_num in range(6):
        print(f"Player No. {test_num}")
        result = result.append(calc_score(test_num, trained_model, use_model_epoch, config_path, plot_results=plot_results,
                                          input_as_random_noize=input_as_random_noize, make_notelvl_from_framelvl=make_notelvl_from_framelvl, verbose=verbose))
    result = result.append(result.describe()[1:3])
    result.to_csv(csv_path, float_format="%.3f")
    return


if __name__ == "__main__":
    main()
