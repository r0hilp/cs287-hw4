#!/usr/bin/env python

"""Word segmentation preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re

def get_char_ids(file_list, dataset=''):
    # Map chars to unique ids
    char_to_idx = {}
    idx = 1
    for file in file_list:
        if file:
            with(open(file, "r")) as f:
                chars = f.read().strip().split()
                for char in chars:
                    if char not in char_to_idx:
                        char_to_idx[char] = idx
                        idx += 1
        f.close()
    return char_to_idx

def convert_data(data_name, char_to_idx):
    char_features = []
    char_output = []

    # Construct input
    with open(data_name, "r") as f:
        chars = f.read().strip().split()
        for char in chars:
            char_features.append(char_to_idx[char])

    # Construct output
    for i in range(len(char_features)-1):
        if char_features[i+1] == char_to_idx['<space>']:
            char_output.append(2)
        else:
            char_output.append(1)
    # EOF
    char_output.append(1)

    return np.array(char_features, dtype=np.int32), np.array(char_output, dtype=np.int32)

FILE_PATHS = {"PTB": ("data/train_chars.txt",
                      "data/valid_chars.txt",
                      "data/test_chars.txt"),
              "PTB_check": ("data/train_check.txt",
                            "data/valid_check.txt",
                            "data/test_check.txt")}
args = {}

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    # parser.add_argument('--backprop_length', help="Backprop length for training", type=int)
    # parser.add_argument('--batch_num', help="Number of batches for training", type=int)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    # backprop_length = args.backprop_length (take as input in training file)
    # batch_num = args.batch_num
    train, valid, test = FILE_PATHS[dataset]

    # Retrive word to id mapping
    print 'Get char ids...'
    char_to_idx = get_char_ids([train, valid, test], dataset=dataset)

    # Convert data
    print 'Processing data...'
    train_input, train_output = convert_data(train, char_to_idx)

    if valid:
        valid_input, valid_output = convert_data(valid, char_to_idx)

    if test:
        test_input, _ = convert_data(test, char_to_idx)

    # Save data
    print 'Saving...'
    filename = dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
        f['vocab_size'] = np.array([len(char_to_idx)], dtype=np.int32)
        f['space_char'] = np.array([char_to_idx['<space>']], dtype=np.int32)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
