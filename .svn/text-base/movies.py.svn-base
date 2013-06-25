import numpy as np

def get_matrix_from_data():
	'''Reads the big data file, returns a nice 943 x 1682 matrix, 
	where rows are users and columns are movies.
	'''
	data_file = open("ml-100k/u.data", 'r')
	matrix = np.zeros((943,1682))
	for line in data_file.readlines():
		line = line.split()
		user_id = int(line[0])
		movie_id = int(line[1])
		rating = int(line[2])
		matrix[user_id-1][movie_id-1] = rating
	return matrix


