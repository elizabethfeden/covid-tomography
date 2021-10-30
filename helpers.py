import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from PIL import Image
from typing import Tuple, Optional

  
def load_datasets_by_names(
    filenames: List[str],
    size: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
  """Returns list of numpy arrays with datasets found by filenames.
  
  Args:
    filenames: the result will be a list of datasets, each representing a
        filename from this list.
    size: if not None, resizes all the images to size.
  """
  datasets = []
  for filename in filenames:
    d =  np.load(filename)
    if size is not None:
      new_shape = [d.shape[3], d.shape[0], *size]
      d_new = np.zeros(new_shape)
      for i in range(d.shape[3]):
        d_new[i] = resize_dataset(d[:, :, :, i], size=size)
      d = d_new
    
    datasets.append(d)
  return datasets


def load_datasets(size=None):
  return load_datasets_by_names([
      'raw-data-kaggle/images_medseg.npy',
      'raw-data-kaggle/masks_medseg.npy',
      'raw-data-kaggle/images_radiopedia.npy',
      'raw-data-kaggle/masks_radiopedia.npy'
      ], size=size)


def resize_dataset(d, size=(256, 256)):
    return np.array([np.array(Image.fromarray(image).resize(size))
                     for image in d])
