import os
from typing import Any, Callable, Dict, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import cv2
from torch.utils.data import WeightedRandomSampler
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")


class FakeDetectorPreprocessor:
    """
    Class for preprocessing the data for the FakeDetector model.
    """
    
    def __init__(self, path: str, 
                 image_size: int = 224,
                 SEED: int = 42,
                 normalize: bool = True) -> None:
        
        """
        Constructor for the FakeDetectorPreprocessor class.

        Args:
            path (str): Path to the dataset.
            image_size (int): Size of the images.
            SEED (int): Seed for the random number generator.
            normalize (bool): Whether to normalize the data.
        """

        if not normalize:
            
            print('Warning: You are not normalizing the data. Make sure to normalize the data before training the model.\
                  If you use a pretrained model, make sure to use `set_transforms` for adding pretrained statistics.')
        
        torch.manual_seed(SEED)

        self.normalize = normalize
        
        self.path = path
        self.image_size = image_size

        self.train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def vizualize_images(self,
                         n: int = 3,
                         fig_size: Tuple[int, int] = (20, 10)) -> None:
        """
        Vizualize the images from the dataset and see number of images in each class.

        Args:
            n (int): Number of images to vizualize for each class.
            fig_size (Tuple[int, int]): Size of the figure.
        """

        number_of_fake_images = len(os.listdir(f'{self.path}/fake'))
        number_of_real_images = len(os.listdir(f'{self.path}/real'))

        fig_title = f'Fake: {number_of_fake_images} | Real: {number_of_real_images}'

        fake_images = [cv2.imread(f'{self.path}/fake/{img}') for img in np.random.choice(os.listdir(f'{self.path}/fake'), n)]
        real_images = [cv2.imread(f'{self.path}/real/{img}') for img in np.random.choice(os.listdir(f'{self.path}/real'), n)]
        
        fig, ax = plt.subplots(2, n, figsize=fig_size)
        fig.suptitle(fig_title, fontsize=16)

        for i in range(n):
            ax[0, i].imshow(cv2.cvtColor(fake_images[i], cv2.COLOR_BGR2RGB))
            ax[0, i].axis('off')
            ax[0, i].set_title('Fake')

            ax[1, i].imshow(cv2.cvtColor(real_images[i], cv2.COLOR_BGR2RGB))
            ax[1, i].axis('off')
            ax[1, i].set_title('Real')

        ax[0, 0].set_ylabel('Fake Images', fontsize=16)
        ax[1, 0].set_ylabel('Real Images', fontsize=16)
        
        plt.show()


    def set_transforms(self, 
                       train_transforms_args: List[Callable],
                       test_transforms_args: List[Callable]) -> None:
        """
        Set the train and test transforms.

        Args:
            train_transforms_args (List[Callable]): List of transforms for the training set.
            test_transforms_args (List[Callable]): List of transforms for the test set.
        """
        
        self.train_transforms = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), 
                                                    *train_transforms_args])

        self.test_transforms = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), 
                                                   *test_transforms_args])
                             
        
    def __compute_statistics(self, 
                           dataset : ImageFolder) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and standard deviation of the dataset.

        Args:
            dataset (ImageFolder): Dataset to compute the statistics for.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation of the dataset, for each channel.
        """
        
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        mean = 0.
        std = 0.
        for images, _ in loader:
            batch_samples = images.size(0)  
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
        mean /= len(loader.dataset)
        std /= len(loader.dataset)
        
        return mean, std

    def prepare_data(self,
                     train_size: float = 0.7,
                     val_size: float = 0.15,
                     n_workers: int = 4,
                     batch_size: int = 64,
                     use_weighted_sampling: bool = False) -> None:
        """
        Prepare the data for training, validation and testing, computing transformations
        and creating the DataLoader objects.

        Args:
            n_workers (int): Number of workers for the DataLoaders.
        """

        dataset = ImageFolder(root=self.path, transform=self.train_transforms)

        train_size = int(train_size * len(dataset))
        val_size = int(val_size * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], 
                                                                generator=torch.Generator().manual_seed(42))
        print(f'Train Size: {len(train_dataset)} | Validation Size: {len(val_dataset)} | Test Size: {len(test_dataset)}')

        if self.normalize:
            print('Computing statistics for normalization...')    
            mean, std = self.__compute_statistics(train_dataset)
            print(f'Statistics (per channel) of the Train Set: Mean: {mean} | Std: {std}')
            self.train_transforms.transforms.append(transforms.Normalize(mean, std))
            self.test_transforms.transforms.append(transforms.Normalize(mean, std))

        val_dataset.dataset.transform = self.test_transforms
        test_dataset.dataset.transform = self.test_transforms

        if use_weighted_sampling:
            train_targets = [dataset.targets[i] for i in train_dataset.indices]
            
            class_sample_counts = [0, 0]  
            for target in train_targets:
                class_sample_counts[target] += 1

            weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
            sample_weights = torch.tensor([weights[target] for target in train_targets])
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=n_workers)
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)        