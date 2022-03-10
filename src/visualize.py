import numpy as np
import librosa
import librosa.display
from matplotlib import lines as mlines, pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
import seaborn as sns
import librosa
import yaml
import os
import argparse
import glob
from multiprocessing import Pool
from itertools import repeat


def plot_tab(tab, note_resolution):
    string_style_dict = {0: 'r', 1: 'y', 2: 'b',
                         3: '#FF7F50', 4: 'g', 5: '#800080'}
    fret_style_dict = {0: '#CC0033', 1: '#3300CC', 2: '#9900CC',
                       3: '#FFCC00', 4: "#00CC99", 5: "#99CC00",
                       6: "#99FF99", 7: "#3366CC", 8: "#006600",
                       9: "#CC9933", 10: "#CC33CC", 11: "#003333",
                       12: "#00CC00", 13: "#999900", 14: "#CC6666",
                       15: "#330066", 16: "#66CCCC", 17: "#663300",
                       18: "#66FFFF", 19: "#9933FF"}
    pitch_style_list = np.array([["#000000", "#CC0000", "#993300", "#FF6600", "#CC9900", "#339900", "#006633", "#009999", "#003399", "#3300CC", 
                                  "#9900CC", "#CC0099", "#000000", "#CC0000", "#993300", "#FF6600", "#CC9900", "#339900", "#006633", "#009999"],
                                 ["#339900", "#006633", "#009999", "#003399", "#3300CC", "#9900CC", "#CC0099", "#000000", "#CC0000", "#993300",
                                     "#FF6600", "#CC9900", "#339900", "#006633", "#009999", "#003399", "#3300CC", "#9900CC", "#CC0099", "#000000"],
                                 ["#9900CC", "#CC0099", "#000000", "#CC0000", "#993300", "#FF6600", "#CC9900", "#339900", "#006633", "#009999",
                                     "#003399", "#3300CC", "#9900CC", "#CC0099", "#000000", "#CC0000", "#993300", "#FF6600", "#CC9900", "#339900"],
                                 ["#FF6600", "#CC9900", "#339900", "#006633", "#009999", "#003399", "#3300CC", "#9900CC", "#CC0099", "#000000",
                                  "#CC0000", "#993300", "#FF6600", "#CC9900", "#339900", "#006633", "#009999", "#003399", "#3300CC", "#9900CC"],
                                 ["#009999", "#003399", "#3300CC", "#9900CC", "#CC0099", "#000000", "#CC0000", "#993300", "#FF6600", "#CC9900",
                                  "#339900", "#006633", "#009999", "#003399", "#3300CC", "#9900CC", "#CC0099", "#000000", "#CC0000", "#993300"],
                                 ["#000000", "#CC0000", "#993300", "#FF6600", "#CC9900", "#339900", "#006633", "#009999", "#003399", "#3300CC", 
                                  "#9900CC", "#CC0099", "#000000", "#CC0000", "#993300", "#FF6600", "#CC9900", "#339900", "#006633", "#009999"]])

    plt.hlines(y=[0,1,2,3,4,5], xmin=0, xmax=64, colors='k', lw=0.15, zorder=0)
    
    for time in range(len(tab)):
        if time % note_resolution == 0:
            plt.vlines(x=time, ymin=0, ymax=5, colors='k', lw=0.3, zorder=0)
        elif time % 4 == 0:
            plt.vlines(x=time, ymin=0, ymax=5, colors='k', lw=0.15, zorder=0)
        else:
            plt.vlines(x=time, ymin=0, ymax=5, colors='k', lw=0.1, ls='dotted', zorder=0)
        for string in range(6):
            fret = np.argmax(tab[time, string])
            if fret != 20:
                plt.scatter(time, string, s=50, marker="${}$".format(
                    fret), color=pitch_style_list[string, fret], linewidths=1, zorder=5)
                
            # plot 'not played' as x marker
            """
            else:
                plt.scatter(time, string, s=20, marker='x',
                            color='black', linewidths=0.5)
            """
    plt.vlines(x=time+1, ymin=0, ymax=5.0, colors='k', lw=0.3)
    plt.xticks([i for i in range(0, time+1) if i % note_resolution == 0],
               labels=[i for i in range(0, time//note_resolution+1)])
    plt.yticks(np.arange(6), ('E', 'A', 'D', 'G', 'B', 'e'))
    plt.ylim(-0.5, 5.5)
    plt.xlabel('Bar number')


def visualize(npz_filename_list, kwargs):
    note_resolution = kwargs["note_resolution"]
    down_sampling_rate = kwargs["down_sampling_rate"]
    bins_per_octave = kwargs["bins_per_octave"]
    hop_length = kwargs["hop_length"]
    encoder_layers = kwargs["encoder_layers"]
    encoder_heads = kwargs["encoder_heads"]
    mode = kwargs["mode"]
    input_feature_type = kwargs["input_feature_type"]
    visualize_dir = kwargs["visualize_dir"]

    npz_filename_list = npz_filename_list.split("\n")

    for npz_filename in npz_filename_list:
        npz_file = np.load(npz_filename)
        # load from saved npz file from src/predict.py
        if mode == "F0":
            input_features = npz_file["input_features"]
            frame_pred = npz_file["frame_F0_pred"]
            frame_gt = npz_file["frame_F0_gt"]
            note_pred = npz_file["note_F0_pred"]
            note_gt = npz_file["note_F0_gt"]
            attn_map = npz_file["attn_map"]
        elif mode == "tab":
            input_features = npz_file["input_features"]
            frame_pred = npz_file["frame_tab_pred"]
            frame_gt = npz_file["frame_tab_gt"]
            note_pred = npz_file["note_tab_pred"]
            note_gt = npz_file["note_tab_gt"]
            frame_F0_from_tab_pred = npz_file["frame_F0_from_tab_pred"]
            frame_F0_gt = npz_file["frame_F0_gt"]
            note_F0_from_tab_pred = npz_file["note_F0_from_tab_pred"]
            note_F0_gt = npz_file["note_F0_gt"]
            attn_map = npz_file["attn_map"]

        # plotting
        frames_per_second = hop_length / down_sampling_rate
        frames_to_sec_labels = np.arange(len(frame_gt)) / frames_per_second
        n_subplots = (
            3 + encoder_layers * encoder_heads) if mode == "F0" else (5 + encoder_layers * encoder_heads)

        plt.figure(figsize=(10, n_subplots*3), dpi=500)
        plt.rc('axes', labelsize=15) 
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        subplot_counter = 1
        plt.subplot(n_subplots, 1, subplot_counter)
        if input_feature_type == "cqt":
            plt.title(f"Input Constant-Q Transform")
            cqt = input_features.T
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(cqt)),
                                     x_axis='time',
                                     y_axis='cqt_hz',
                                     sr=down_sampling_rate,
                                     hop_length=hop_length,
                                     bins_per_octave=bins_per_octave,
                                     cmap='magma')
        elif input_feature_type == "melspec":
            plt.title(f"Input Mel-spectrogram")
            melspec = input_features.T
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(melspec)),
                                     x_axis='time',
                                     y_axis='hz',
                                     sr=down_sampling_rate,
                                     hop_length=hop_length,
                                     cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Tims [s]')
        subplot_counter = subplot_counter + 1

        if mode == "F0":
            # frame level prediction
            plt.subplot(n_subplots, 1, subplot_counter)
            plt.title('Frame-level F0')
            librosa.display.specshow((frame_gt * 0.4 + frame_pred).T,
                                     x_axis='time',
                                     y_axis=None,
                                     sr=down_sampling_rate,
                                     hop_length=hop_length,
                                     cmap='hot')
            plt.yticks([8, 20, 32], ['C3', 'C4', 'C5'])
            plt.ylabel('pitch')
            TP_patch = mpatches.Patch(color='white', label='TP')
            FP_patch = mpatches.Patch(color='yellow', label='FP')
            FN_patch = mpatches.Patch(color='red', label='FN')
            plt.legend(handles=[TP_patch, FP_patch, FN_patch])
            plt.xlabel('Tims [s]')
            subplot_counter = subplot_counter + 1

            # note level F0 prediction
            plt.subplot(n_subplots, 1, subplot_counter)
            plt.title('Note-level F0')
            sns.heatmap((note_gt * 0.4 + note_pred).T, cmap='hot', cbar=False,
                        rasterized=False).invert_yaxis()
            plt.legend(handles=[TP_patch, FP_patch, FN_patch])
            for n_note in range(len(note_gt)+1):
                if n_note % 16 == 0:
                    plt.axvline(n_note, color='white', lw=1)
                else:
                    plt.axvline(n_note, color='white', lw=0.2)
            plt.xticks([i for i in range(0, n_note) if i % note_resolution == 0],
                       labels=[i for i in range(0, n_note//note_resolution+1)])
            plt.yticks([8, 20, 32], ['C3', 'C4', 'C5'])
            plt.xlabel('Bar number')
            plt.ylabel('pitch')
            subplot_counter = subplot_counter + 1

        elif mode == "tab":
            plt.subplot(n_subplots, 1, subplot_counter)
            plt.title('Ground truth note-level tablature')
            plot_tab(note_gt, note_resolution)
            subplot_counter = subplot_counter + 1

            plt.subplot(n_subplots, 1, subplot_counter)
            plt.title('Predicted note-level tablature')
            plot_tab(note_pred, note_resolution)
            subplot_counter = subplot_counter + 1

            # frame level F0 converted from tab prediction
            plt.subplot(n_subplots, 1, subplot_counter)
            plt.title('Frame-level F0 converted from Tab estimation')
            librosa.display.specshow((frame_F0_gt * 0.4 + frame_F0_from_tab_pred).T,
                                     x_axis='time',
                                     y_axis=None,
                                     sr=down_sampling_rate,
                                     hop_length=hop_length,
                                     cmap='hot')
            plt.yticks([8, 20, 32], ['C3', 'C4', 'C5'])
            plt.ylabel('pitch')
            TP_patch = mpatches.Patch(color='white', label='TP')
            FP_patch = mpatches.Patch(color='yellow', label='FP')
            FN_patch = mpatches.Patch(color='red', label='FN')
            plt.legend(handles=[TP_patch, FP_patch, FN_patch])
            plt.xlabel('Tims [s]')
            subplot_counter = subplot_counter + 1

            # note level F0 converted from tab prediction
            plt.subplot(n_subplots, 1, subplot_counter)
            plt.title('Note-level F0 converted fron Tab estimation')
            sns.heatmap((note_F0_gt * 0.4 + note_F0_from_tab_pred).T,
                        cmap='hot', cbar=False, rasterized=False).invert_yaxis()
            plt.legend(handles=[TP_patch, FP_patch, FN_patch])
            for n_note in range(len(note_F0_gt)+1):
                if n_note % 16 == 0:
                    plt.axvline(n_note, color='white', lw=1)
                else:
                    plt.axvline(n_note, color='white', lw=0.2)
            plt.xticks([i for i in range(0, n_note) if i % note_resolution == 0],
                       labels=[i for i in range(0, n_note//note_resolution+1)])
            plt.yticks([8, 20, 32], ['C3', 'C4', 'C5'])
            plt.xlabel('Bar number')
            plt.ylabel('pitch')
            subplot_counter = subplot_counter + 1
        
        # encoder self-attention
        for n_layer in range(encoder_layers):
            for n_head in range(encoder_heads):
                plt.subplot(n_subplots, 1, subplot_counter)
                plt.title(f'Encoder self-attention map')
                cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
                sns.heatmap(attn_map[n_layer, n_head], cmap=cmap, 
                            norm=LogNorm(vmin=1e-3))
                plt.xlabel('source sequence')
                plt.ylabel('target sequence')
                subplot_counter = subplot_counter + 1
        
        plt.tight_layout()
        save_filename = os.path.join(
            visualize_dir,  f"{os.path.split(npz_filename)[1][:-4]}.png")
        if os.path.exists(save_filename):
            os.remove(save_filename)
        plt.savefig(save_filename)
        plt.close('all')
        
        print(f"finished {os.path.split(npz_filename)[1][:-4]}")


def main():
    parser = argparse.ArgumentParser(description='code for plotting results')
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

    config_path = os.path.join("model", f"{trained_model}", "config.yaml")
    with open(config_path) as f:
        obj = yaml.safe_load(f)
        note_resolution = obj["note_resolution"]
        down_sampling_rate = obj["down_sampling_rate"]
        bins_per_octave = obj["bins_per_octave"]
        hop_length = obj["hop_length"]
        encoder_layers = obj["encoder_layers"]
        encoder_heads = obj["encoder_heads"]
        n_cores = obj["n_cores"]
        mode = obj["mode"]
        input_feature_type = obj["input_feature_type"]

    kwargs = {
        "note_resolution": note_resolution,
        "down_sampling_rate": down_sampling_rate,
        "bins_per_octave": bins_per_octave,
        "hop_length": hop_length,
        "encoder_layers": encoder_layers,
        "encoder_heads": encoder_heads,
        "mode": mode,
        "input_feature_type": input_feature_type
    }

    if mode == "F0":
        npz_dir = os.path.join(
            "result", "F0", f"{trained_model}_epoch{use_model_epoch}", "npz")
    elif mode == "tab":
        npz_dir = os.path.join(
            "result", "tab", f"{trained_model}_epoch{use_model_epoch}", "npz")

    for test_num in range(6):
        if mode == "F0":
            visualize_dir = os.path.join(
                "result", "F0", f"{trained_model}_epoch{use_model_epoch}", "visualize", f"test_0{test_num}")
        if mode == "tab":
            visualize_dir = os.path.join(
                "result", "tab", f"{trained_model}_epoch{use_model_epoch}", "visualize", f"test_0{test_num}")

        npz_filename_list = glob.glob(
            os.path.join(npz_dir, f"test_0{test_num}", "*"))
        kwargs["visualize_dir"] = visualize_dir
        if not(os.path.exists(visualize_dir)):
            os.makedirs(visualize_dir)
            
        # paralell process
        p = Pool(n_cores)
        p.starmap(visualize, zip(npz_filename_list, repeat(kwargs)))
        p.close()  # or p.terminate()
        p.join()


if __name__ == "__main__":
    main()
