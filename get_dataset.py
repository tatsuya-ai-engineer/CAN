import torch
from torchvision import datasets
from torchvision.transforms import transforms


def get_cifar10(batch_size, dataloader_workers, dataset_directory="./data"):
    # Prepare dataset for training
    data_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root=dataset_directory, train=True,
                               download=True, transform=data_transformation)

    # Use sampler for randomization
    sampler = torch.utils.data.SubsetRandomSampler(range(len(dataset)))

    # Prepare Data Loaders for training and validation
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                              pin_memory=True, num_workers=dataloader_workers)

    return dataset, data_loader