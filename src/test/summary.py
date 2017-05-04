'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
Author: Yang Shuai
Project: https://github.com/ys10/TensorFlowDemo
'''

from __future__ import print_function

import time, os
import configparser
import logging
import tensorflow as tf
import h5py
import math
from tensorflow.contrib import rnn
from src.lstm.utils import *

try:
    from tensorflow.python.ops import ctc_ops
except ImportError:
    from tensorflow.contrib.ctc import ctc_ops


# Import configuration by config parser.
cp = configparser.ConfigParser()
cp.read('../../conf/test/summary.ini')

# Config the logger.
# Output into log file.
log_file_name = cp.get('log', 'log_dir') + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.log'
if not os.path.exists(log_file_name):
    f = open(log_file_name, 'w')
    f.close()
logging.basicConfig(level = logging.DEBUG,
                format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=log_file_name,
                filemode='w')
# Output to the console.
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Import data set
# Name of file storing trunk names.
train_names_file_name = cp.get('data', 'train_names_file_name')
# Read trunk names.
train_names_file = open(train_names_file_name, 'r')

# Import data set
# Name of file storing trunk names.
dev_names_file_name = cp.get('data', 'dev_names_file_name')
# Read trunk names.
dev_names_file = open(dev_names_file_name, 'r')
#
train_lines = train_names_file.readlines();
dev_lines = dev_names_file.readlines();
train_num = len(train_lines);
dev_num = len(dev_lines);
#
line_in = 0;
for line in dev_lines:
    if train_lines.__contains__(line):
        line_in += 1;
#
logging.debug("train_num: " + str(train_num))
logging.debug("dev_num: " + str(dev_num))
logging.debug("line_in: " + str(line_in))
