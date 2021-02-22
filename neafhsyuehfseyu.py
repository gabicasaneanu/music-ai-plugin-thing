
import tensorflow as tf
from tensorflow.keras import regularizers
import re
import functools
import array
import os
import librosa
import librosa.display
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import matplotlib.pyplot as plt
import numpy

def ker_setter():
    return [ 
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                       input_shape=(9, 13, 1),
                                       kernel_regularizer=regularizers.l2(0.001)),
                tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu',
                                      kernel_regularizer=regularizers.l2(0.001)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(len(train_path_list)),
                ]

def sound_aug():
    if 'np_mfcc.npy' in os.listdir():
     
        smp = load('np_mfcc.npy')
        sample_list = load('sample_list.npy')
    else:
        for i in range(len(arr)):  
    
          
            sound = AudioSegment.from_file(arr[i], format='wav', channels=1)
            sound = sound.set_channels(1)
    
            pitched_sounds = []
            augmentor(sound, aug)
    
            kcs(i, pitched_sounds, sample_list, smp)
    
            if i % 100 == 0:
                print (np.floor(i * 100 / len(arr)))
        save('np_mfcc.npy', smp)
        sample_list = np.array(sample_list)
        save('sample_list.npy', sample_list)
    return smp, sample_list

def __main__():
    def dir_set():
        def __dir__():
            class_names = ['kick', 'snare', 'clap', 'hihat']
            train_path_list = ['kicks\\', 'snares\\', 'claps\\',
                               'hats\\']
            test_path_list = ['kicks\\', 'snares\\', 'claps\\',
                              'hats\\']
            return class_names, test_path_list, train_path_list
        
        class_names, test_path_list, train_path_list = __dir__()
        
        def __directory__():
            arr = []
            for a in train_path_list:
                for i in os.listdir(a):
                    arr.append(a + i)
            return arr
        
        arr = __directory__()
        
        
        sample_list = []
        
        n_fft = 2048
        hop_length = 512
        #ptc
        sample_length = 100 
        augumentor = 9
        return __dir__, __directory__, arr, augumentor, hop_length, n_fft, sample_length
    
    __dir__, __directory__, arr, augumentor, hop_length, n_fft, sample_length = dir_set()
    
    entry_amount = len(arr) * augumentor
    smp = np.empty((entry_amount, 9, 13))
    
    def lowpass(cutoff, sound):
        lp_filter = AudioSegment.low_pass_filter(sound, cutoff * 10 + 100)
        augmented_sound = sound + lowpassed
        return augmented_sound
    
    
    def pitchSample(octaves, sound):
        new_sample_rate = int(sound.frame_rate * 2.0 ** octaves)
        hipitch_sound = sound._spawn(sound.raw_data,
                                     overrides={'frame_rate': new_sample_rate})
        hipitch_sound = hipitch_sound.set_frame_rate(44100)
        return hipitch_sound
    
    
    
    def ms_samples(sample_length):
        return int(44100 / 1000 * sample_length)
    
    
    def augmentor(sound, aug):
        if aug == 1:
            pitched_sounds.append(sound)
        else:
            for i in range(aug):
                aug = float(aug)
                spread = aug / 100 - aug * 2 / 100 + aug / 100 * i
                pitched_sounds.append(pitchSample(spread, sound))
    
               
    
        return pitched_sounds
    
    
    
    
    
    def padding_sm(samples):
        if len(samples) < ms_samples(sample_length):
            padding_samples = ms_samples(sample_length) \
                - len(samples)
            for dumi in range(padding_samples):
                samples.append(0)
        return samples
    
    def dir2_search(i, pitched_sounds, sample_list, x):
        if re.search('kicks', arr[i]):
            sample_list.append(0)
        elif re.search('snares', arr[i]):
            sample_list.append(1)
        elif re.search('clap', arr[i]):
            sample_list.append(2)
        else:
            sample_list.append(3)
        
        sound = (pitched_sounds[x])[:sample_length]
        
        samples = sound.get_array_of_samples()
        
        return padding_sm(samples)
    
    def kcs(i, pitched_sounds, sample_list, smp):
        for x in range(len(pitched_sounds)):
        
            samples = dir2_search(i, pitched_sounds, sample_list, x)
        
            
        
            samples = np.array(samples)
            samples = samples.astype(float)
            mfcc = librosa.feature.mfcc(samples, n_fft=n_fft,
                    hop_length=hop_length, n_mfcc=13)
            mfcc = mfcc.T
        
            
            mfcc = np.expand_dims(mfcc, axis=0)
        
            smp[i * aug + x] = mfcc
    
    def sound_factor():
        return sound_aug()
    
    smp, sample_list = sound_factor()
    
    
    
    smp = np.expand_dims(smp, axis=3)
    
    
    
    def seed_shuff():
        seed = 10
        np.random.seed(seed)
        np.random.shuffle(smp)
        np.random.seed(seed)
        np.random.shuffle(sample_list)
        np.random.seed()
    
    def seed_sample():
        seed_shuff()
        
        
        
        arr = []
        for a in test_path_list:
            for i in os.listdir(a):
                arr.append(a + i)
        
        test_sample_list = []
        
        
        
        test_np_mfcc = np.empty((len(arr), 9, 13))
        return a, arr, test_np_mfcc, test_sample_list
    
    a, arr, test_np_mfcc, test_sample_list = seed_sample()
    
    def dir_search(i):
        if re.search('kicks', arr[i]):
            test_sample_list.append(0)
        elif re.search('snares', arr[i]):
            test_sample_list.append(1)
        elif re.search('claps', arr[i]):
            test_sample_list.append(2)
        else:
            test_sample_list.append(3)
    
    def sample_mfcc(i, samples):
        samples = np.array(samples)
        samples = samples.astype(float)
        mfcc = librosa.feature.mfcc(samples, n_fft=n_fft,
                                    hop_length=hop_length, n_mfcc=13)
        mfcc = mfcc.T
        
        mfcc = np.expand_dims(mfcc, axis=0)
        test_np_mfcc[i] = mfcc
        return samples
    
    def smp_padding(i, samples):
        if len(samples) < ms_samples(sample_length):
            padding_samples = ms_samples(sample_length) - len(samples)
            for padno in range(padding_samples):
                samples.append(0)
        
        
        
        return sample_mfcc(i, samples)
    
    def SoundSegments():
        for i in range(len(arr)):
        
        
            sound = AudioSegment.from_file(arr[i], format='wav', channels=1)
            sound = sound.set_channels(1)
        
            sound = sound[:sample_length]
            samples = sound.get_array_of_samples()
        
            dir_search(i)
        
            samples = smp_padding(i, samples)
    
    SoundSegments()
    
    test_sample_list = np.array(test_sample_list)
    test_np_mfcc = np.expand_dims(test_np_mfcc, axis=3)
    
    
    
    def quick_shuffle():
        seed = 10
        np.random.seed(seed)
        np.random.shuffle(test_np_mfcc)
        np.random.seed(seed)
        np.random.shuffle(test_sample_list)
        np.random.seed(seed)
        np.random.shuffle(arr)
        np.random.seed()
        return seed
    
    def train_m():
        train = int(amount_entries * 0.9)
        val_size = amount_entries - train
        
        training_ds = tf.data.Dataset.from_tensor_slices((smp, sample_list))
        
        
        val_ds = training_ds.skip(train).take(val_size)
        training_ds = training_ds.take(train)
        
        print (val_ds)
        print (training_ds)
        return train, training_ds, val_ds, val_size
    
    def init_keras_values():
        seed = quick_shuffle()
        
        train, training_ds, val_ds, val_size = train_m()
        
        
        
        batch_size = 320
        STEPS_PER_EPOCH = train // batch_size
        return STEPS_PER_EPOCH, batch_size, seed, train, training_ds, val_ds, val_size
    
    STEPS_PER_EPOCH, batch_size, seed, train, training_ds, val_ds, val_size = init_keras_values()
    
    
    def get_callbacks():
        return [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                patience=6)]  
    
    def lr_schedule():
        return tf.keras.optimizers.schedules.InverseTimeDecay(0.001,
               decay_steps=STEPS_PER_EPOCH * 1000, decay_rate=1,
               staircase=False)
    
    lr_schedule = lr_schedule()
    
    
    def get_optimizer():
        return tf.keras.optimizers.Adam(lr_schedule)
    
    
    inputShape = (9, 13, 1)
    
    def k_m():
        model = tf.keras.models.Sequential(ker_setter())
        return model
    
    def ker_model():
        model = k_m()
        
        loss_fn = \
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])  
        
        history = model.fit(training_ds.shuffle(train).batch(batch_size),
                            epochs=40,
                            validation_data=val_ds.batch(batch_size),
                            callbacks=get_callbacks(), verbose=1)
        return history, loss_fn, model
    
    history, loss_fn, model = ker_model()  
    
        
    
    model.evaluate(test_np_mfcc, test_sample_list, verbose=2)
    
    
    
    probability_model = tf.keras.Sequential([model,
            tf.keras.layers.Softmax()])
    
    def class_mod():
        for i in range(len(test_np_mfcc)):
            result = probability_model(test_np_mfcc[i:i + 1])  
            answer = np.argmax(result[0])
            print (arr[i] + ' is a ' + class_names[answer])
        return i
    
    i = class_mod()
    
    
    def lbl(predictions_array, true_label):
        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        
        plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label],
                   100 * np.max(predictions_array),
                   class_names[true_label]), color=color)
    
    def plot_image(
        i,
        predictions_array,
        true_label,
        img,
        ):
        (predictions_array, true_label, img) = (predictions_array,
                true_label[i], img[i])
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    
        plt.plot(img)
    
        lbl(predictions_array, true_label)
    
    
    def plot_value_array(i, predictions_array, true_label):
        (predictions_array, true_label) = (predictions_array, true_label[i])
        plt.grid(False)
        plt.xticks(range(4))
        plt.yticks([])
        thisplot = plt.bar(range(4), predictions_array, color='#777777')
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
    
        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')
    return SoundSegments, augmentor, class_mod, dir2_search, dir_search, dir_set, entry_amount, get_callbacks, get_optimizer, i, init_keras_values, inputShape, k_m, kcs, ker_model, lbl, lowpass, lr_schedule, ms_samples, padding_sm, pitchSample, plot_image, plot_value_array, probability_model, quick_shuffle, sample_mfcc, seed_sample, seed_shuff, smp, smp_padding, sound_factor, test_np_mfcc, test_sample_list, train_m

SoundSegments, augmentor, class_mod, dir2_search, dir_search, dir_set, entry_amount, get_callbacks, get_optimizer, i, init_keras_values, inputShape, k_m, kcs, ker_model, lbl, lowpass, lr_schedule, ms_samples, padding_sm, pitchSample, plot_image, plot_value_array, probability_model, quick_shuffle, sample_mfcc, seed_sample, seed_shuff, smp, smp_padding, sound_factor, test_np_mfcc, test_sample_list, train_m = __main__()
