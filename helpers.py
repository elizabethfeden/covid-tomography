import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional


DATASET_NAMES = [
      Path('raw-data-kaggle/images_medseg.npy'),
      Path('raw-data-kaggle/masks_medseg.npy'),
      Path('raw-data-kaggle/images_radiopedia.npy'),
      Path('raw-data-kaggle/masks_radiopedia.npy'),
]
LABELS = ['Ground Glass', 'Consolidation', 'Lungs Other', 'Background']

  
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
    d = np.load(filename)
    if size is not None:
      new_shape = [d.shape[3], d.shape[0], *size]
      d_new = np.zeros(new_shape)
      for i in range(d.shape[3]):
        d_new[i] = resize_dataset(d[:, :, :, i], size=size)
      d = d_new
    else:
      d = d.reshape(np.array(d.shape)[[3, 0, 1, 2]])
    
    datasets.append(d)
    print(f'Added {filename}, shape {d.shape}')
  return datasets


def load_datasets(prefix_path: str,
                  size: Optional[Tuple[int, int]] = None
                  ) -> List[np.ndarray]:
  return load_datasets_by_names(
      [Path(prefix_path) / name for name in DATASET_NAMES],
      size=size)


def resize_dataset(dataset: np.ndarray,
                   size: Tuple[int, int] = (256, 256)
                   ) -> np.ndarray:
    """Resizes all the images in dataset."""
    return np.array([np.array(Image.fromarray(image).resize(size))
                     for image in dataset])

def show_im_row(images: np.ndarray,
                size: int = 10,
                titles: Optional[List[str]] = None
                ) -> None:
    """Shows images in a row."""
    n = images.shape[0]
    if titles is None:
        titles = [''] * n
    figure, axes = plt.subplots(1, n, figsize=(n * size, size))
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap='Greys')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
def plot_loss(model) -> None:
    """Draws loss plot from model's `loss_history` parameter."""
    plt.plot(range(len(model.loss_history)), model.loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

