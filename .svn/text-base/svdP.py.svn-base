import numpy as np
import eigP
import genA
import math
import random
import sys
import matplotlib.pyplot as plt
import movies 

def svdP(k, eps, A):
    '''Computes the Singular Value Decomposition of the matrix A,
    using k singular values, and converging with an eps bound.

    Returns a tuple, containing a list of left singular vectors, a list of
    singular values, and a list of right singular vectors.
    '''
    results = eigP.eigP(k, eps, np.dot(A.T, A))
    right_singulars = [vec for (vec, val, c) in results]
    singular_vals = map(math.sqrt, [val for (vec, val, c) in results])

    left_singulars = []
    for i in range(k):
        dot_prod = np.dot(A, right_singulars[i])
        vec = np.divide(dot_prod, np.linalg.norm(dot_prod))
        left_singulars.append(vec)

    generate_output(left_singulars, right_singulars, singular_vals)
    return (left_singulars, singular_vals, right_singulars)


def reconstruct(left_singulars, singular_vals, right_singulars):
    '''Constructs the matrix from a list of left singular vectors,
    a list of singular values, and a list of right singular vectors.
    '''
    ls = np.array(left_singulars).T
    sv = np.diag(singular_vals)
    rs = np.array(right_singulars).T
    return np.dot(ls, np.dot(sv, rs.T))

def generate_output(left_singulars, right_singulars, singular_vals):
    '''Writes the output to an ASCII file:
    left_singulars.txt: The left singular vectors
    right_singulars.txt: The right singular vectors
    singular_vals.txt: The singular values
    '''
    left_singulars_file = open("left_singulars.txt", 'w')
    right_singulars_file = open("right_singulars.txt", 'w')
    singular_vals_file = open("singular_vals.txt", 'w')

    left_sing_array = np.array(left_singulars).T
    for row in left_sing_array:
        left_singulars_file.writelines([str(val) + " " for val in row])
        left_singulars_file.write('\n')

    right_sing_array = np.array(right_singulars).T
    for row in right_sing_array:
        right_singulars_file.writelines([str(val) + " " for val in row])
        right_singulars_file.write('\n')

    singular_vals_file.writelines([str(val) + '\n' for val in singular_vals])


def get_matrix(file):
    '''Returns a numpy matrix object from the ASCII file'''
    array = np.genfromtxt(file)
    return array

def generate_test_matrix():
    '''Generates a 10 x 20 test matrix'''
    min_val = -100
    max_val = 100
    vals = [[random.uniform(min_val, max_val) for _ in range(20)] for _ in range(10)]
    return np.array(vals)

def frobenius_test(eps):
    ''' Tests the similarity between a random matrix, and its
    reconstruction using SVD.
    '''
    random_matrix = generate_test_matrix()
    square_norms = []
    for k in xrange(1, 11):
        (ls, sv, rs) = svdP(k, eps, random_matrix)
        restruct = reconstruct(ls, sv, rs)
        diff = abs(random_matrix - restruct)
        square_norm = np.linalg.norm(diff)**2
        square_norms.append(square_norm)

    
    plt.plot(list(xrange(1,11)), square_norms)
    plt.xlabel("Number of Singular Values")
    plt.ylabel("Squared Frobenius Norm")
    plt.show()

def test_with_missing_entries(eps, k=10, p=.2):
    '''Tests SVD by trying to reconstruct a matrix 
    that is missing entries.
    '''
    (A, A_bar) = genA.genA(k, p)
    (ls, sv, rs) = svdP(k, eps, A_bar)
    restruct = reconstruct(ls, sv, rs)
    diff = abs(A - restruct)

    return np.linalg.norm(diff)**2

def plot_k(eps):
    '''Plots the square of Frobenius norm against the number
    of singular values used.
    '''
    ks = list(xrange(1,31,2))
    p = 0.9

    median_squared_norms = []
    for k in ks:
        squared_norms = []
        for i in range(50):
            print i
            norm = test_with_missing_entries(eps, k, p)
            squared_norms.append(norm)
        median_squared_norms.append(np.median(squared_norms))

    plt.plot(ks, median_squared_norms)
    plt.xlabel("Number of singular values")
    plt.ylabel("Square of Frobenius norm")
    plt.show()

def plot_p(eps):
    '''Plots the square of Frobenius norm against the Probability
    that an entry appears in the matrix.'''
    k = 5
    ps = [.05*i for i in xrange(1, 20)]

    median_squared_norms = []
    for p in ps:
        squared_norms = []
        for i in range(50):
            norm = test_with_missing_entries(eps, k, p)
            squared_norms.append(norm)
        median_squared_norms.append(np.median(squared_norms))

    plt.plot(ps, median_squared_norms)
    plt.xlabel("Probability of an entry appearing")
    plt.ylabel("Square of Frobenius norm")
    plt.show()

def analyze_movies():
    '''Analyzes the accuracy of SVD with increasing k'''
    movie_matrix = movies.get_matrix_from_data()

    # Test reconstruction accuracy
    ks = [5+5*i for i in range(20)]
    norms = []
    for k in ks:
        print k
        (ls, sv, rs) = svdP(k, .01, movie_matrix)
        recon = reconstruct(ls, sv, rs)
        diff = abs(movie_matrix - recon)
        norms.append(np.linalg.norm(diff)**2)

    plt.plot(ks, norms)
    plt.xlabel("Number of singular values")
    plt.ylabel("Square of Frobenius Norm")
    plt.show()

def pick_out_genres():
    '''Used for picking out genres.
    Shows the index and weight corresponding to the highest weighted movies
    of each eigenvector.
    '''
    movie_matrix = movies.get_matrix_from_data()

    (ls, sv, rs) = svdP(5, .01, movie_matrix)
    for vec in rs:
        tups = []
        it = np.nditer(vec, flags=['f_index'])
        while not it.finished:
            tups.append((it[0], it.index))
            it.iternext()
        print sorted(tups, reverse=True)[:10] # Best movies in each genre



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Invalid number of arguments. Proper use:"
        print "> python eigP.py <k> <epsilon> <matrix_file_path>"
        print "k = number of eigenvectors"
        print "epsilon = bound for power method"
        print "matrix_file_path is a path to an ASCII file encoding a symmetric matrix"
        raise Exception("Invalid arguments!")
    k = int(sys.argv[1])
    eps = float(sys.argv[2])
    matrix_filepath = sys.argv[3]
    matrix_file = open(matrix_filepath, 'r')
    matrix = get_matrix(matrix_file)
    svdP(k, eps, matrix)
    print "SVD successful! Look at left_singulars.txt, singular_vals.txt, and right_singulars.txt to see results!"


