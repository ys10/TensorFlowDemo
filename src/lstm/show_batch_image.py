import h5py

# File storing group name.
group_file_name = "../../tmp/data/ctc/train_speechorder_half-swbd_ctc.txt"
# Read group data.
groups = open(group_file_name, 'r');
# Name of HDF5 file as test data set.
test_data_file_name = "../../tmp/data/ctc/train_half-swbd_ctc.hdf5"
# Read training data file.
test_data = h5py.File(test_data_file_name, 'r')
# # Name of HDF5 file as result data set.
result_data_file_name = "../../tmp/data/ctc/train_half-swbd_ctc.hdf5"
# Read result data file.
result_data = h5py.File(result_data_file_name, 'r')


