# +
import xmltodict
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import shutil
import pandas as pd
import tqdm


def plot_ecg(waveform, save_path: Path):
    fig, axs = plt.subplots(12, figsize=(6, 6), sharex=True)
    for i in range(waveform.shape[0]):
        axs[i].plot(waveform[i])
        plt.savefig(save_path, facecolor="white")
        plt.close()


xml_source = Path("/path/to/xmls")
output = Path("/path/to/output")

if output.exists():
    shutil.rmtree(output)
output.mkdir()

# set to true to attempt to generate plots of what this code is doing to the waveforms.
do_plot = True
has_plotted = False
num_saved = 0
num_discarded = 0
num_without_rhythm = 0

filelist = os.listdir(xml_source)

with tqdm(len(filelist), force_tty=True) as bar:
    for fname in filelist:
        if has_plotted:
            do_plot = False

        try:
            # using a library called xmltodict to interpret the XML files as python dicts
            with open(xml_source / fname, "r") as f:
                xml_dict = xmltodict.parse(f.read())["ECGSCAN_ECG"]
        except:
            print(f"{fname} could not be opened.")
            num_discarded += 1
            continue
        leads = []
        starts = []
        ends = []
        clip_lengths = []

        # took some trial and error to figure out where in the dict
        # the actual waveform data was stored, but eventually found it
        for i, lead in enumerate(xml_dict["MASH"]):
            wave_xml = xml_dict["MASH"][lead]
            y_values = np.array([float(y) for y in wave_xml["YPOS"].split(" ")])
            leads.append(y_values)
            clip_lengths.append(len(y_values))

            x_values = np.array([int(x) for x in wave_xml["XPOS"].split(" ")])
            starts.append(min(x_values))
            ends.append(max(x_values))

        # skipping any files that are missing a lead or have fewer than 12 leads
        if any([x == 0 for x in clip_lengths]) or len(leads) < 12:
            print(f"{fname} is missing at least 1 lead")
            num_discarded += 1
            continue

        # for fields that inexplicably have more than 12 leads, keep only the first 12
        leads = leads[:12]

        start = min(starts)
        end = max(ends) + 1
        recording_length = end - start

        # very rough method for detecting a common structure that we observed in some of these files
        # where there would be a bunch of channels that had ~2.5 seconds of data in them, and then one
        # channel that would have a full 10 seconds. Since we don't know what temporal resolution any of
        # these are at, I just looked for files where one of the channels was at least 3x as long as the
        # rest of the channels
        longest_clip_length = max(clip_lengths)
        if not (any(x < longest_clip_length / 3 for x in clip_lengths)):
            num_without_rhythm += 1

        if do_plot:
            plot_ecg(np.stack(leads), "pngs/og.png")
        # Somewhat obtuse code for lining up all of the channels' starts and then scaling them all
        # so that they're at least 1250 in length, then cutting off everything after 1250.
        blank = np.zeros((12, 5000))

        for i, lead in enumerate(leads):
            basis = np.zeros((recording_length,))
            if len(lead) > recording_length:
                lead = lead[:recording_length]

            percentage = len(lead) / recording_length
            if percentage < 0.25:
                lead = np.interp(
                    np.linspace(0, len(lead), int(recording_length * 0.26)),
                    np.arange(len(lead)),
                    lead,
                )

            basis[: len(lead)] = lead
            resampled = np.interp(
                np.linspace(0, recording_length, 5000),
                np.arange(recording_length),
                basis,
            )

            blank[i] = resampled

        if do_plot:
            plot_ecg(np.stack(leads), "pngs/resampled.png")
        # final input size to the model is 12 x 1250
        waveforms = blank[:, :1250]

        npy_fname = fname + ".npy"
        # Save the waveform as a numpy array
        # np.save(output/npy_fname, waveforms)
        num_saved += 1
        if do_plot:
            plot_ecg(waveforms, "pngs/trimmed")
            has_plotted = True
        bar.text = f"D: {num_discarded}, NR: {num_without_rhythm}"
        bar()

print(f"Saved: {num_saved}, Discarded: {num_discarded}, Total: {len(filelist)}")
