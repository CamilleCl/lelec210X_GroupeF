import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf

from classification.datasets import Dataset, get_cls_from_path
from classification.utils.plots import plot_audio, plot_specgram

# read sounds
dataset = Dataset()

sound = dataset["helicopter", 0]
x, fs = sf.read(sound)

print(sound)

print(f'Playing a "{get_cls_from_path(sound)}"')

sd.play(x, fs)