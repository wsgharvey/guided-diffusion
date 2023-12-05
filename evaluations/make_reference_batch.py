#%%

import numpy as np
import PIL.Image
import os
import zipfile
import argparse
from pathlib import Path

#%%

parser = argparse.ArgumentParser()
parser.add_argument(
    'path',
     type=str,
     help='e.g. ../../edm/datasets/ffhq-64x64.zip',
)
path = Path(parser.parse_args().path)

#%%

# open .zip image dataset
z = zipfile.ZipFile(path)
fnames = z.namelist()

# stack all of these images into a single array
arr_0 = []
for fname in fnames:
    if not fname.endswith('.png'):
        continue
    with z.open(fname, 'r') as f:
        image = np.array(PIL.Image.open(f))
        arr_0.append(image)

arr_0 = np.stack(arr_0, axis=0)

# %%

# print(arr_0.shape)
save_path = Path('.') / (path.stem + '-mini' + '.npz')
print(f'Saving dataset file with shape {arr_0.shape} to {save_path}')
np.savez(save_path, arr_0=arr_0[:100])

exit()
#%%

np.load(save_path)['arr_0'].shape

# %%

