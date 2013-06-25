import numpy as np
import sys
import random

def generate_random_matrix(k):
	'''Generates a random matrix using k singular values'''     
	left_sing_vals = [map(np.linalg.norm, \
		[random.gauss(0,1) for _ in range(k)]) for _ in range(30)]

	right_sing_vals = [map(np.linalg.norm, \
		[random.gauss(0,1) for _ in range(30)]) for _ in range(k)]

	sing_vals = [random.uniform(0,1) for _ in range(k)]
	# sing_vals.extend([0 for _ in range(30-k)])

	U = np.array(left_sing_vals)
	V = np.array(right_sing_vals)
	S = np.diag(sing_vals)

	return np.dot(U, np.dot(S, V))

def remove_vals_from_matrix(matrix, p):
	'''Removes values from the matrix with probablity 1-p.
	Uses the uniform distribution to do this.
	'''
	for row in range(matrix.shape[0]):
		for col in range(matrix.shape[1]):
			sample_num = random.uniform(0,1)
			if sample_num >= p:
				matrix[row][col] = 0
	return matrix

def genA(k, p):
	'''Returns (A, A_bar), where A is a random matrix, and 
	A_bar is the same matrix with its entries removed with a probability 
	1-p.
	'''
	A = generate_random_matrix(k)
	A_copy = np.copy(A)
	A_bar = remove_vals_from_matrix(A_copy, p)
	
	return (A, A_bar)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Invalid number of arguments. Proper use:"
        print "> python eigP.py <k> <p>"
        print "k = number of eigenvectors"
        print "p = probability that each entry appears"
        raise Exception("Invalid Arguments!")
    k = int(sys.argv[1])
    p = float(sys.argv[2])

    (A, A_bar) = genA(k, p)
    print "### Matrix A ###"
    print A
    print "### Same matrix with some entries removed ###"
    print A_bar

    print "success!"



