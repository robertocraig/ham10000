import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
from torchvision import transforms

import numpy as np
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

path = '/content/drive/MyDrive/PUCRS/Ap Profundo 2/HAM10000/skin_df.csv'
img_dir = '/content/drive/MyDrive/PUCRS/Ap Profundo 2/HAM10000/zip/all_images'

class SkinDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Initializes the SkinDataset.

        Args:
            csv_file (str): Path to the CSV file with annotations. The CSV should include columns such as
                            'lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization', 'image_path'.
            img_dir (str): Directory where all the images are stored.
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        """
        #self.data_frame = pd.read_csv(csv_file)

        if isinstance(csv_file, str):
            self.data_frame = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            self.data_frame = csv_file
        else:
            raise ValueError("csv_file must be either a file path (str) or a DataFrame.")

        self.img_dir = img_dir
        self.transform = transform

        # Mapping from short labels to full names and to numeric labels
        self.label_mapping = {
            'nv': {'name': 'nevus', 'numeric': 0},
            'mel': {'name': 'melanoma', 'numeric': 1},
            'bkl': {'name': 'benign keratosis-like lesion', 'numeric': 2},
            'bcc': {'name': 'basal cell carcinoma', 'numeric': 3},
            'akiec': {'name': 'actinic keratosis', 'numeric': 4},
            'vasc': {'name': 'vascular lesion', 'numeric': 5},
            'df': {'name': 'dermatofibroma', 'numeric': 6}
        }

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, index):
      """
      Retrieves the image, label, and metadata for a given index.

      Args:
        index (int): Index of the item to fetch.

      Returns:
        dict: A dictionary with the following keys:
            - 'image': The transformed image (if transform is provided) or the original PIL image.
            - 'label': The numeric label of the image (int).
            - 'full_label': The full descriptive label of the image (str).
            - 'metadata': A dictionary containing additional metadata (age, sex, localization, dx_type).

      Raises:
        IndexError: If the provided index is out of range.
        FileNotFoundError: If the image file is not found at the specified path.
      """
      if index < 0 or index >= len(self.data_frame):
        raise IndexError(f"Index {index} is out of range for the dataset of length {len(self.data_frame)}.")

      # Get the image path from the DataFrame
      img_path = self.data_frame.iloc[index]['image_path']

      # Load the image from the file path
      try:
        image = Image.open(img_path)
      except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found at path: {img_path}")

      # Apply any transformations if provided
      if self.transform:
        image = self.transform(image)
      else:
        # Caso transform seja None, converte manualmente para tensor
        image = transforms.ToTensor()(image)

      # Converts image back to PIL image
      #if isinstance(image, torch.Tensor):
      #  image = transforms.ToPILImage()(image)

      # Get the label
      label = self.data_frame.iloc[index]['dx']
      full_label = self.label_mapping[label]['name']
      #numeric_label = self.label_mapping[label]['numeric']
      numeric_label = torch.tensor(self.label_mapping[label]['numeric'])  # Convert label to tensor

      # Additional metadata (age, sex, localization, dx_type)
      metadata = {
        'age': self.data_frame.iloc[index]['age'],
        'sex': self.data_frame.iloc[index]['sex'],
        'localization': self.data_frame.iloc[index]['localization'],
        'dx_type': self.data_frame.iloc[index]['dx_type']
      }

      # Return a dictionary containing the image, its label, and metadata
      return {'image': image, 'label': label, 'full_label': full_label, 'numeric_label': numeric_label, 'metadata': metadata, 'index': index}

    def get_labels(self):
        """
        Returns all shorthand labels in the dataset.

        Returns:
            list: A list of shorthand labels (e.g., ['nv', 'mel', ...]).
        """
        return list(self.data_frame['dx'])

    def get_full_labels(self):
        """
        Returns all labels in the dataset as full descriptive names.

        Returns:
            list: A list of full descriptive labels (e.g., ['nevus', 'melanoma', ...]).
        """
        return [self.label_mapping[label]['name'] for label in self.data_frame['dx']]

    def get_label(self, index):
        """
        Returns the full descriptive label for a given index.

        Args:
            index (int): Index of the item to fetch the label for.

        Returns:
            str: The full descriptive label (e.g., 'melanoma').

        Raises:
            IndexError: If the provided index is out of range.
        """
        if index < 0 or index >= len(self.data_frame):
            raise IndexError(f"Index {index} is out of range for the dataset of length {len(self.data_frame)}.")

        short_label = self.data_frame.iloc[index]['dx']
        return self.label_mapping[short_label]['name']

    def get_image_name(self, index):
        """
        Returns the image file name for a given index.

        Args:
            index (int): Index of the item to fetch the image name for.

        Returns:
            str: The image file name (e.g., 'ISIC_0015719').

        Raises:
            IndexError: If the provided index is out of range.
        """
        if index < 0 or index >= len(self.data_frame):
            raise IndexError(f"Index {index} is out of range for the dataset of length {len(self.data_frame)}.")

        return self.data_frame.iloc[index]['image_id']

    def get_metadata(self, index):
        """
        Returns all metadata for a given index as a dictionary.

        Args:
            index (int): Index of the item to fetch metadata for.

        Returns:
            dict: A dictionary containing all metadata for the given index, including 'lesion_id',
                  'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization', and 'image_path'.

        Raises:
            IndexError: If the provided index is out of range.
        """
        if index < 0 or index >= len(self.data_frame):
            raise IndexError(f"Index {index} is out of range for the dataset of length {len(self.data_frame)}.")

        return self.data_frame.iloc[index].to_dict()

    def get_label_distribution(self):
      """
      Returns the distribution of labels in the dataset.

      Returns:
        dict: A dictionary where the keys are the full descriptive labels, and the values are the counts of each label.
      """
      label_counts = self.data_frame['dx'].value_counts()
      return {self.label_mapping[label]['name']: count for label, count in label_counts.items()}

    def get_samples_by_label(self, label, max_samples=None):
      """
      Returns a subset of samples that match a given label.

      Args:
        label (str): The full descriptive label (e.g., 'melanoma').
        max_samples (int, optional): The maximum number of samples to return. If None, returns all samples.

      Returns:
        list: A list of dictionaries, where each dictionary contains 'image', 'label', 'full_label', and 'metadata'.
      """
      short_label = next((k for k, v in self.label_mapping.items() if v['name'] == label), None)
      if short_label is None:
          raise ValueError(f"Label '{label}' is not a valid label.")

      filtered_data = self.data_frame[self.data_frame['dx'] == short_label]

      if max_samples:
          filtered_data = filtered_data.head(max_samples)

      return [self[i] for i in filtered_data.index]

    def visualize_sample(self, index):
      """
      Displays the image and metadata for a given index.

      Args:
        index (int): Index of the item to visualize.

      Raises:
        IndexError: If the provided index is out of range.
      """
      if index < 0 or index >= len(self.data_frame):
          raise IndexError(f"Index {index} is out of range for the dataset of length {len(self.data_frame)}.")

      sample = self[index]
      image = sample['image']

      # Convert tensor to PIL image if necessary
      if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

      plt.figure(figsize=(8, 6))  # Width, height in inches
      plt.title(f"Label: {sample['full_label']}\nAge: {sample['metadata']['age']} | Sex: {sample['metadata']['sex']} | Localization: {sample['metadata']['localization']} | Index: {index}", fontsize=12)
      plt.imshow(image)
      plt.axis('off')  # Hide axes
      plt.tight_layout()
      plt.show()

    def visualize_by_lesion_id(self, lesion_id):
      """
      Displays the original image and metadata for a given lesion_id.

      Args:
        lesion_id (str): Lesion ID of the image to visualize.

      Raises:
        ValueError: If the lesion_id is not found in the dataset.
      """
      # Verificar se o lesion_id existe no DataFrame
      lesion_row = self.data_frame[self.data_frame['lesion_id'] == lesion_id]
    
      if lesion_row.empty:
        raise ValueError(f"Lesion ID {lesion_id} not found in the dataset.")
    
      # Pegar o índice da primeira ocorrência do lesion_id
      index = lesion_row.index[0]
    
      # Obter o caminho original da imagem
      img_path = lesion_row.iloc[0]['image_path']
    
      # Carregar a imagem original diretamente do arquivo
      original_image = Image.open(img_path)

      # Obter as demais informações (rótulo, metadata) para exibição
      full_label = self.label_mapping[lesion_row.iloc[0]['dx']]['name']
      metadata = {
        'age': lesion_row.iloc[0]['age'],
        'sex': lesion_row.iloc[0]['sex'],
        'localization': lesion_row.iloc[0]['localization']
      }

      # Exibir a imagem original
      plt.figure(figsize=(8, 6))  # Configura o tamanho da figura
      plt.title(f"Original Image - Lesion ID: {lesion_id}\nLabel: {full_label}\nAge: {metadata['age']} | Sex: {metadata['sex']} | Localization: {metadata['localization']}", fontsize=12)
      plt.imshow(original_image)
      plt.axis('off')  # Ocultar eixos
      plt.tight_layout()
      plt.show()
    
    def augment_data(self, augmentations):
      """
      Applies data augmentation techniques to the dataset.

      Args:
          augmentations (callable): A function or a list of transformations to apply to each image.

      Returns:
          SkinDataset: A new SkinDataset instance with the augmentations applied.
      """
      augmented_transform = transforms.Compose([self.transform, augmentations]) if self.transform else augmentations
      return SkinDataset(csv_file=self.data_frame.copy(), img_dir=self.img_dir, transform=augmented_transform)


import os
import multiprocessing

# Get the number of CPU cores
num_workers = multiprocessing.cpu_count()

def data_load(dataset, batch_size=32, shuffle=True, val_size=0.2, test_size=0.1, num_workers=4):
    """
    Loads train, val, and test data using PyTorch's DataLoader.
    
    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int, optional): Number of samples per batch to load. Default is 32.
        shuffle (bool, optional): Whether to shuffle the training data at every epoch. Default is True.
        val_size (float, optional): Proportion of the dataset to include in the validation split. Default is 0.2.
        test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.1.
        num_workers (int, optional): Number of subprocesses to use for data loading. Default is 4.
    
    Returns:
        tuple: A tuple containing the DataLoader for the training, validation, and test data.
    """
    # Calculate split sizes
    total_size = len(dataset)
    test_split = int(test_size * total_size)
    val_split = int(val_size * total_size)
    train_split = total_size - test_split - val_split

    # Split the dataset into training, validation, and testing sets
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        range(total_size),
        [train_split, val_split, test_split]
    )
    
    # Subset the dataset based on train, val, and test indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create DataLoaders for train, val, and test datasets
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader