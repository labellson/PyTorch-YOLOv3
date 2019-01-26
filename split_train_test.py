from torch.utils import data
from utils.datasets import LabelboxDataset
from utils.parse_config import parse_data_config
from utils.utils import load_classes
from os import path

import os

from argparse import ArgumentParser


def convert_dataset(dataset, dataset_name):
    for i in range(len(dataset)):
        img_path, _, labels = dataset[i]
        img_base_path, img_name = path.split(img_path)

        dataset_path = path.join('/', *img_base_path.split('/')[:-1],
                                 dataset_name)

        if not path.exists(dataset_path):
            os.mkdir(dataset_path)

        # Create symlinks to the original images
        dest_path = path.join(dataset_path, img_name)
        if not path.exists(dest_path):
            os.symlink(img_path, dest_path)

        # Create the .txt file with the images path
        with open(path.join(path.dirname(path.dirname(dataset_path)),
                            dataset_name + '.txt'), 'a+') as txt_file:
            txt_file.write('{}\n'.format(dest_path))

        # Create the labels .txt
        label_path = dest_path.replace('images', 'labels').rsplit('.')[0] + '.txt'
        if not path.exists(path.dirname(label_path)):
            os.mkdir(path.dirname(label_path))

        with open(label_path, 'w+') as label_txt:
            for class_, cx, cy, w, h in labels:
                label_txt.write('{:d} {:f} {:f} {:f} {:f}\n'.format(int(class_),
                                                                    cx, cy, w, h))

        

ag = ArgumentParser()
ag.add_argument('--data_config_path', type=str, default='./config/epi.data',
                help='path to data config path')
ag.add_argument('--class_path', type=str, default='./data/epi.names',
                help='path to class label file')
ag.add_argument('--test_split', type=float, default=.2,
                help='percentage test split')

args = ag.parse_args()
classes = load_classes(args.class_path)
data_config = parse_data_config(args.data_config_path)
labels_path = data_config['data']
train_path = data_config['train']
test_path = data_config['test']
test_split = args.test_split

# Open the dataset without padding
dataset = LabelboxDataset(labels_path, classes, padding=False)

# Split the dataset
n_test = int(len(dataset) * test_split)
n_train = len(dataset) - n_test
train_set, test_set = data.random_split(dataset, (n_train, n_test))

# Loop and save the coordinates in .txt files
# Make test and train folders with symbolic links
convert_dataset(train_set, 'train')
convert_dataset(test_set, 'test')
