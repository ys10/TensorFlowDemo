import configparser
import logging
import time, os
import h5py
# import tensorflow as tf
# from src.lstm.utils import sparse_tuple_from as sparse_tuple_from

# Import configuration by config parser.
cp = configparser.ConfigParser()
cp.read('../../conf/test/statistic.ini')


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
trunk_names_file_name = cp.get('data', 'trunk_names_file_name')
# Name of HDF5 file as training data set.
data_file_name = cp.get('data', 'data_file_name')
# Read trunk names.
trunk_names_file = open(trunk_names_file_name, 'r')
# Read training data set.
test_data_file = h5py.File(data_file_name, 'r')

n_classes = 47 # total classes

# y = tf.sparse_placeholder(tf.int32, [None])

result = [];
for i in range(n_classes):
   result.append(0);

def one_hot_to_index (one_hot):
    for i in range(0, len(one_hot), 1):
        if one_hot[i] == 1:
            return i
    return -1

# Read all trunk names.
all_trunk_names = trunk_names_file.readlines()
start = time.time()
# decode_grp = iter_grp.create_group("decode")
# Break out of the training iteration while there is no trunk usable.
if not all_trunk_names:
    logging.warning("all_trunk_names not exist!")
trunk_name = ''
# Every batch only contains one trunk.
trunk = 0
for line in all_trunk_names:
    trunk_name = line.split()[1]
    # trunk_name = line.split()[0]
    print("trunk_name: " + trunk_name)
    # print("length:"+ len(trunk_name))
    # Define two variables to store input data.
    # decode = []
    # Get trunk data by trunk name without line break character.
    # sentence_y is a tensor of shape (None)
    decode = None
    try:
        decode = test_data_file['iter0/decode/' + trunk_name.strip('\n')]
    except KeyError:
        continue
    # Add current trunk into the batch.
    # batch_y.append(sentence_y)
    # batch_y = sparse_tuple_from(batch_y)
    # logging.debug(decode.value[0])
    # break;
    for phome in decode.value[0]:
        # logging.debug("phome:" + str(int(phome)))
        result[int(phome)] += 1
    logging.debug(
        "Trunk:" + str(trunk) + " name:" + str(trunk_name) + ", time = {:.3f}".format(time.time() - start))
    logging.debug(result)
    trunk += 1
    # break;
logging.debug("trunk:"+str(trunk))
