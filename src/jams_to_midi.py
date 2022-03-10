import jams
import numpy as np
import os
import glob
import jams_interpreter
import pretty_midi
import tqdm
import yaml


def main(note_resolution):
    jams_file_path = os.path.join("GuitarSet", "annotation", "*")
    jams_filename_list = glob.glob(jams_file_path)
    jams_filename_list.sort()
    midi_dir = os.path.join("data", "midi")
    midi_dir_original = os.path.join(midi_dir, "original")
    midi_dir_quantized = os.path.join(
        midi_dir, f"auto_quantized_{note_resolution}")

    if not(os.path.exists(midi_dir)):
        os.makedirs(midi_dir_original)
        os.makedirs(midi_dir_quantized)

    for jams_filename in tqdm.tqdm(jams_filename_list):
        jams_file = jams.load(jams_filename)
        tempo = float(jams_filename.split('-')[1])
        midi_file_original = jams_interpreter.jams_to_midi(
            jams_file, tempo=tempo, q=0)
        midi_file_quantized = jams_interpreter.jams_to_midi(
            jams_file, tempo=tempo, q=0, quantization=note_resolution)
        midi_filename_original = os.path.join(
            midi_dir_original, os.path.split(jams_filename)[1][:-5])
        midi_filename_quantized = os.path.join(
            midi_dir_quantized, os.path.split(jams_filename)[1][:-5])
        midi_file_original.write(midi_filename_original + ".mid")
        midi_file_quantized.write(midi_filename_quantized + ".mid")

    visualize_path = os.path.join("visualize")
    if not(os.path.exists(visualize_path)):
        os.makedirs(visualize_path)


if __name__ == "__main__":
    with open("src/config.yaml") as f:
        obj = yaml.safe_load(f)
        note_resolution = obj["note_resolution"]
    main(note_resolution)
