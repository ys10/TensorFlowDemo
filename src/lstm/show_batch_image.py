import time, os, math
from PIL import Image
import numpy as np
import h5py
import configparser
import logging
from src.lstm.utils import pad_sequences
from src.lstm.utils import sparse_tuple_from

# Import configuration by config parser.
cp = configparser.ConfigParser()
cp.read('../../conf/ctc/show.ini')

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

# File storing group name.
group_file_name = cp.get('data', 'group_file_name')
# Read group data.
groups = open(group_file_name, 'r');
# Name of HDF5 file as test data set.
test_data_file_name = cp.get('data', 'test_data_file_name')
# Read training data file.
test_data = h5py.File(test_data_file_name, 'r')
# Name of HDF5 file as result data set.
result_data_file_name = cp.get('data', 'result_data_file_name')
# Read result data file.
result_data = h5py.File(result_data_file_name, 'r')
# Image result directory.
image_dir = cp.get('img', 'image_dir')

# Parameters
batch_size = 1
display_batch = 1
training_iters = 1
n_steps = 777 # time steps

all_trunk_names = groups.readlines()
for iter in range(0, training_iters, 1):
    # For each iteration.
    logging.debug("Iter:" + str(iter))
    # Break out of the training iteration while there is no trunk usable.
    if not all_trunk_names:
        break
    logging.debug("number of trunks:" + str(len(all_trunk_names)))
    # Calculate how many batches can the data set be divided into.
    n_batches = math.floor(len(all_trunk_names) / batch_size)
    # n_batches = math.ceil(len(all_trunk_names) / batch_size)
    logging.debug("n_batches:" + str(n_batches))
    # Train the RNN(LSTM) model by batch.
    for batch in range(0, n_batches, 1):
        # For each batch.
        # Define two variables to store input data.
        batch_x = []
        batch_y = []
        batch_seq_len = []
        batch_lstm_outputs = []
        batch_linear_outputs = []
        batch_decode = []
        # Traverse all trunks of a batch.
        for trunk in range(0, batch_size, 1):
            start = time.time()
            trunk_name_index = batch * batch_size + trunk
            logging.debug("trunk_name_index: " + str(trunk_name_index))
            # There is a fact that if the number of all trunks
            #   can not be divisible by batch size,
            #   then the last batch can not get enough trunks of batch size.
            # The above fact is equivalent to the fact
            #   that there is at least a trunk
            #   whose index is no less than the number of all trunks.
            if (trunk_name_index >= len(all_trunk_names)):
                # So some used trunks should be add to the last batch when the "fact" happened.
                # Select the last trunk to be added into the last batch.
                trunk_name_index = len(all_trunk_names) - 1
                logging.info("trunk_name_index >= len(all_trunk_names), trunk_name_index is:" + str(
                    trunk_name_index) + "len(all_trunk_names):" + str(len(all_trunk_names)))
            # Get trunk name from all trunk names by trunk name index.
            trunk_name = all_trunk_names[trunk_name_index]
            logging.debug("trunk_name: " + trunk_name)
            # print("trunk_name: " + trunk_name)
            # print("length:"+ len(trunk_name))
            # Get trunk data by trunk name without line break character.
            # sentence_x is a tensor of shape (n_steps, n_inputs)
            sentence_x = test_data['source/' + trunk_name.strip('\n')]
            # sentence_y is a tensor of shape (None)
            sentence_y = test_data['target/' + trunk_name.strip('\n')]
            # sentence_len is a tensor of shape (None)
            sentence_len = test_data['size/' + trunk_name.strip('\n')].value
            # Add current trunk into the batch.
            batch_x.append(sentence_x)
            batch_y.append(sentence_y)
            #
            linear_outputs = result_data['iter0/linear_output/' + trunk_name.strip('\n')]
            lstm_outputs = result_data['iter0/lstm_output/' + trunk_name.strip('\n')]
            decode = result_data['iter0/decode/' + trunk_name.strip('\n')]
            #
            batch_lstm_outputs.append(lstm_outputs)
            batch_linear_outputs.append(linear_outputs)
            batch_decode.append(decode)
        #
        batch_x, batch_seq_len = pad_sequences(batch_x, maxlen=n_steps)
        batch_y = sparse_tuple_from(batch_y)
        # transfer the result to picture.
        img_label = Image.fromarray(np.uint8(np.array(batch_y.value)))
        img_linear_outputs = Image.fromarray(np.uint8(np.array(batch_linear_outputs.value)))
        img_lstm_outputs = Image.fromarray(np.uint8(np.array(batch_lstm_outputs.value)))
        # show the result by picture.
        img_label.save(image_dir + "img_label" + trunk_name + ".png")
        img_linear_outputs.save(image_dir + "img_linear_outputs" + trunk_name + ".png")
        img_lstm_outputs.save(image_dir + "img_lstm_outputs" + trunk_name + ".png")
        # TODO
        logging.debug("Trunk: " + str(trunk) + " name:" + str(trunk_name))
        logging.debug("label: " + str(batch_y))
        logging.debug("linear_outputs: " + str(linear_outputs))
        logging.debug("lstm_outputs: " + str(lstm_outputs))
        logging.debug("Decode: " + str(decode))
        logging.debug("time: {:.3f}".format(time.time() - start))
        break;
