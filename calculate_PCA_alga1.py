import numpy as np
import os
from scipy import linalg as la
import h5py

## This script does PCA analysis on the 2d ptychography dataset D with dimension J x N
# J: The number of probe positions
# N: The number of pixels in one diffraction patterns generated at each probe position
# The output of the analysis tells you how to factorize D into PR
# P: reduced data matrix
# R: encoding matrix

# the folder and the file of the ptychography dataset;
# the dataset was saved in the dimension: (1, n_probe_pos_y*n_probe_pos_x, n_pixels_y, n_pixels_x)
data_folder = "./"
data_file = "adorym_fly0444_rotate_90_size128_99_180.h5"

# the path of saving the encoding matrix (R)
encoding_save_folder = "algae1_PCA_99_180"
encoding_file = "encoding_matrix.npy"
if 'ref' in data_file: encoding_file = os.path.splitext(encoding_file)[0]+"_ref"+os.path.splitext(encoding_file)[1]
encoding_save_path = os.path.join(encoding_save_folder, encoding_file)

# the path of saveing the weighting matrix (P)
weighting_matrix_save_folder = "algae1_PCA_99_180"
weighting_matrix_file = "weighting_matrix.npy"
if 'ref' in data_file: weighting_matrix_file = os.path.splitext(weighting_matrix_file)[0]+"_ref"+os.path.splitext(weighting_matrix_file)[1]
weighting_matrix_save_path = os.path.join(weighting_matrix_save_folder, weighting_matrix_file)

print("Loading data ... ")
with h5py.File(os.path.join(data_folder, data_file), "r") as f:
    data = f['exchange/data'][...]
    
# Converting the data to intensity (for simulated data)    
# data = data.real[0]**2

# Just the the real part (for experimental data)
data = data.real[0]

# Reshape the dataset to the dimention JxN
data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))

print("Calculate covariance of the data matrix ... ")
# Calculate the covariance matrix and the weighting matrix
Cov = np.cov(data)

print("Calculate eigenvalues and eigenvectors of the covariance matrix ... ")
evals, evecs = la.eigh(Cov)
# sort eigenvalues in decreasing order, using the same order to sort the eigenvectors
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:,idx]

print("Calculate the encoding matrix ... ")
# weighting_matrix(P) = evecs, calculate the encoding_matrix (R) = P^T D 
encoding_matrix = np.matmul(evecs.T, data)

print("Saving matrices ... ")
# ## save the eigenvalues
np.save(os.path.join(weighting_matrix_save_folder, "PCA_eigenvalues.npy"), evals)
# ## save the weighting matrix
np.save(weighting_matrix_save_path, evecs)
## save the encoding matrix
np.save(encoding_save_path, encoding_matrix)
