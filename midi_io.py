from midi.utils import midiread, midiwrite
import matplotlib.pyplot as plt
import numpy as np
import os

# Compatible with Python 2.7. not 3.x
# Recursively load midi files, extract piano rolls and save as *.mid (file) and *.npy (matrix)

load_root = './MIDI_Data/'
save_root = './MIDI_Data_PianoRolls/'

for dirpath, dirs, files in os.walk(load_root):
    for name in files:
        if name.endswith('.mid'):
            print dirpath, name
            print dirpath.replace(load_root,save_root), name
            
            load_dirpath = dirpath
            save_dirpath = dirpath.replace(load_root,save_root)
            
            load_filepath = os.path.join(load_dirpath,name)
            save_filepath = os.path.join(save_dirpath,name)

            # Read MIDI file
            piano_roll = midiread(load_filepath).piano_roll
            
            if not os.path.exists(save_dirpath):
                os.makedirs(save_dirpath)
            
            # Save the piano roll as MIDI
            midiwrite(save_filepath, piano_roll=piano_roll)
            
            # Save the piano roll as *.npy file
            np.save(save_filepath.replace('.mid','.npy'), piano_roll)  
