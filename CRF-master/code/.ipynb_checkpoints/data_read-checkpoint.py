import numpy

def read_decode_input():
#function to read the data from decode_input
#100 letters each 128 elements
#26 weight vectors each 128 elements
#T which is a 26 x 26 weight matrix
#T is row major T_11, T_21, T_31 ..

	with open("../data/decode_input.txt", "r") as f:
		raw_data = f.read().split("\n")

	X = numpy.array(raw_data[:100*128], dtype=float).reshape(100,128)
	W = numpy.array(raw_data[100*128:100*128+26*128]\
		, dtype=float).reshape(26,128)
	T = numpy.array(raw_data[100*128+26*128:-1], dtype=float).reshape(26,26)
	T = numpy.swapaxes(T, 0, 1)

	return X, W, T

def read_train_struct():  #contemplate modifying this for performance reasons
#function to read data into list structure
#dataX number of examples by 128 currently a numpy array
#dataY number of examples by 2 "array" (each example has a label and a qid)

	with open("../data/train_struct.txt", "r") as f:
		raw_data = f.read()
	raw_data = raw_data.split("\n")

	dataX, dataY = [], []
	for line in raw_data[:-1]: #-2 because last element is empty
		line = line.split(" ")
		dataY.append([ int(line[0])-1, int(line[1][4:]) ])
		datax = [0]*128
		for f1 in line[2:]:
			end = f1.find(":")
			datax[int(f1[:end])-1] = 1
		dataX.append(datax)

	return numpy.array(dataX, dtype=float), dataY

def read_model():
#function to read model for 2a
	with open("../data/model.txt", "r") as f:
		raw_data = f.read()
	raw_data = raw_data.split("\n")

	W = numpy.array(raw_data[:26*128], dtype=float).reshape(26, 128)
	T = numpy.array(raw_data[26*128:-1], dtype=float).reshape(26, 26)
	T = numpy.swapaxes(T, 0, 1)
	return W, T

def read_train():
#function to read train data
	from string import ascii_lowercase
	mapping = list(enumerate(ascii_lowercase))
	mapping = { i[1]:i[0] for i in mapping }

	with open("../data/train.txt", "r") as f:
		raw_data = f.read()
	raw_data = raw_data.split("\n")

	dataX, dataY = [], []
	tempX, tempY = [], []
	for row in raw_data[:-1]:
		row = row.split(" ")
		tempY.append( mapping[row[1]])
		tempX.append( numpy.array(row[5:], dtype=float) )
		if int(row[2]) < 0:
			dataX.append(numpy.array(tempX))
			dataY.append(numpy.array(tempY, dtype=int))
			tempX, tempY = [], []

	ret = zip(dataX, dataY)
	return list(ret)

def read_test():
#function to read train data
	from string import ascii_lowercase
	mapping = list(enumerate(ascii_lowercase))
	mapping = { i[1]:i[0] for i in mapping }

	with open("../data/test.txt", "r") as f:
		raw_data = f.read()
	raw_data = raw_data.split("\n")

	dataX, dataY = [], []
	tempX, tempY = [], []
	for row in raw_data[:-1]:
		row = row.split(" ")
		tempY.append( mapping[row[1]])
		tempX.append( numpy.array(row[5:], dtype=float) )
		if int(row[2]) < 0:
			dataX.append(numpy.array(tempX))
			dataY.append(numpy.array(tempY, dtype=int))
			tempX, tempY = [], []

	ret = zip(dataX, dataY)
	return list(ret)

def read_test_decoder_modified(file_path):
    """
    Reads the test data for decoding and returns a NumPy array
    where each sub-array from the list becomes a row in the final
    two-dimensional array.
    """
    import numpy as np

    # Read the file
    with open(file_path, 'r') as file:
        raw_data = file.read().strip().split('\n')

    # Split each line into its components and extract the features
    dataX = [list(map(float, row.split(' ')[5:]))
             for row in raw_data if row]

    # Convert the list of lists (features for each word) into a 2D NumPy array
    dataX_np = np.array(dataX)

    return dataX_np



#This function is similar to your read_test function but without the dataY part, since we only need the X data (features) for decoding #purposes. You might need to adjust the path to "../data/test.txt" depending on where your test data file is located relative to the script #that is calling this function.
