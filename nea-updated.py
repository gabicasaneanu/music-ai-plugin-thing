
import os
from pydub import AudioSegment
from pydub.playback import play
import numpy as np 
import tensorflow as tf
from tensorflow.keras import regularizers
import re
import functools
print = functools.partial(print, flush=True)
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import save
from numpy import load
import librosa, librosa.display





class_names = ["kick","snare","clap","hihat"] 

train_path_list = ["train_kicks\\","train_snares\\","train_claps\\","train_hats\\"]
test_path_list = ["test_kicks\\","test_snares\\","test_claps\\","test_hats\\"]

arr = []
for a in train_path_list:
	for i in os.listdir(a):
		arr.append(a  + i)


print(len(arr))

sample_list = []

n_fft = 2048
hop_length = 512


def pitchSample(octaves,sound):
	new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
	hipitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
	hipitch_sound = hipitch_sound.set_frame_rate(44100)
	return hipitch_sound

def bassBoostSample(cutoff,sound):
	print("cutoff: ",cutoff*10)
	lowpassed = AudioSegment.low_pass_filter(sound,(cutoff*10)+100)
	augmented_sound = sound + lowpassed
	return augmented_sound

def ms_samples(sample_length):
	return int((44100 / 1000) * sample_length)


def augmentor(sound,aug):
	if aug == 1:
		pitched_sounds.append(sound)
	else:
		for i in range(aug):
			aug = float(aug)
			spread = ((aug/100) - (aug*2)/100) + (aug/100)*i
			pitched_sounds.append(pitchSample(spread,sound))
			# pitched_sounds.append(bassBoostSample(spread,sound))
	return pitched_sounds


sample_length = 100 
aug = 9



amount_entries = len(arr)*aug
np_mfcc = np.empty((amount_entries, 9, 13))





