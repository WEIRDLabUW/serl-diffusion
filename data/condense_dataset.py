import h5py
import numpy as np
import tqdm

# Open the existing HDF5 file in read mode
with h5py.File('image.hdf5', 'r') as old_file:
    # Open a new HDF5 file in write mode
    with h5py.File('condensed_images.hdf5', 'w') as new_file:
        # Iterate over the keys in the 'data' group of the old file
        for key in tqdm.tqdm(old_file['data'].keys()):
            # Copy the 'actions' and 'agentview_image' datasets to the new file
            old_file.copy(f'data/{key}/actions', new_file, f'data/{key}/actions')
            old_file.copy(f'data/{key}/obs/agentview_image', new_file, f'data/{key}/obs/agentview_image')
            old_file.copy(f'data/{key}/obs/agentview_image', new_file, f'data/{key}/obs/agentview_image')