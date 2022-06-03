import numpy as np
import os
import h5py

encoding_read_folder = "algae1_PCA_99_180"
encoding_file = "encoding_matrix.npy"
encoding_read_path = os.path.join(encoding_read_folder, encoding_file)

data_folder = "./"
data_file = "adorym_fly0444_rotate_90_size128_99_180.h5"
data_path = os.path.join(data_folder, data_file)

# probe steps
Np_x = 141
Np_y = 161

# Diffraction pattern size
N_x = 128
N_y = 128


PCA_output_folder = "algae1_PCA_99_180"
if not os.path.exists(PCA_output_folder):
    os.makedirs(PCA_output_folder)
    
mse_error_ls = []
n_limit_ls = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
# n_limit_ls = [1500, 2000, 2500]

# read the data, data array dimension = (n_theta, Np_y * Np_x, N_y, N_x)
with h5py.File(data_path, "r") as f:
    data = f['exchange/data'][...] 
    
# because the dataset is for a 2D sample, n_theta=1
data = data[0]

# reshape the dataset to the dimention JxN
data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))    

## load the encoding matrix (R_(SxN))
encoding_matrix = np.load(encoding_read_path)

for n_limit in n_limit_ls:
    print("S' = ", n_limit)

    ## reduce the encoding matrices based on n_limit. R_(SxN) reduces to R_(S'xN)
    reduced_encoding_matrix = encoding_matrix[:n_limit,:]

    # calculate the inverse of the reduced encoding matrix, R_(NxS')^(-1), using Moore-Penrose inverse
    inv_encoding_matrix = np.linalg.pinv(reduced_encoding_matrix)

    ## calculate the weighting_matrix (namely the compressed data), P_(JxS') = D_(JxN) R_(NxS')^(-1)
    weighting_matrix = np.matmul(data, inv_encoding_matrix[:, :n_limit])

    ## approximately recover the data matrix using the compressed data and the reduced encoding matrix. D ~ P_(JxS') R_(S'xN)
    data_approx = np.matmul(weighting_matrix, reduced_encoding_matrix)
    data_approx = np.clip(data_approx, 0, np.inf)
    mse_error_ls.append(np.sum((data - data_approx)**2) / (Np_y * Np_x * N_y * N_x))
    
    ## reshape the data matrix to match what's required by the reconstruction program
    data_approx = np.reshape(data_approx, (1, Np_y * Np_x, N_y, N_x))
    
    if 'ref' in data_file:
        n_limit = str(n_limit) + '_ref'
    else:
        n_limit = str(n_limit)
    
    with h5py.File(os.path.join(PCA_output_folder, "algae1_size128_nei" + n_limit + ".h5"),"w") as f:
        grp = f.create_group("exchange")
        dsest = grp.create_dataset("data", (1, Np_y * Np_x, N_y, N_x), dtype="f4")

    with h5py.File(os.path.join(PCA_output_folder, "algae1_size128_nei" + n_limit + ".h5"),"r+") as f:
        dset = f["exchange/data"]
        dset[...] = data_approx

if 'ref' in data_file:
    np.savez(os.path.join(PCA_output_folder, "mse_dataset_ref.npz"), n_limit_ls, np.array(mse_error_ls))

else:
    np.savez(os.path.join(PCA_output_folder, "mse_dataset.npz"), n_limit_ls, np.array(mse_error_ls))
