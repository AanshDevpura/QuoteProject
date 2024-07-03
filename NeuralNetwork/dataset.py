from torch.utils.data import Dataset
import torch

class ImageCaptionDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        self.split = split
        
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        self.images = torch.load('data/images.pt')
        self.encoded_captions = torch.load('data/encoded_captions.pt')

        # Determine indices based on split
        if self.split == 'TRAIN':
            start_idx = 0
            end_idx = int(0.98 * len(self.images))
        elif self.split == 'VAL':
            start_idx = int(0.98 * len(self.images))
            end_idx = int(0.99 * len(self.images))
        else:  # TEST
            start_idx = int(0.99 * len(self.images))
            end_idx = len(self.images)

        self.images = self.images[start_idx:end_idx]
        self.encoded_captions = self.encoded_captions[start_idx * 5:end_idx * 5]
        self.data_size = len(self.encoded_captions)

    def __getitem__(self, i):
        image = torch.FloatTensor(self.images[i // 5] / 255.)
        caption = self.encoded_captions[i]

        if self.transform is not None:
            image = self.transform(image)
        return image, caption

    def __len__(self):
        return self.data_size