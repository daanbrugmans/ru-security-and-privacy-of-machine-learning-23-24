import torch
from PIL import Image
from .Utils import *


class MyDataLoader:

    def __init__(self, all_data, indices, batch_size):
        assert all_data is not None
        assert indices is not None
        assert batch_size is not None
        data_x = [all_data.data[x] for x in indices]
        data_x = [MyDataLoader.transform_images(x, all_data) for x in data_x]
        data_x = torch.stack(data_x, axis=0)
        data_y = [all_data.targets[x] for x in indices]
        if isinstance(data_y, list):
            data_y = torch.tensor(data_y)
        total_number_of_samples_for_this_client = len(indices)
        batches_X = batchify(data_x, batch_size,
                             total_number_of_samples_for_this_client)

        batches_y = batchify(data_y, batch_size,
                             total_number_of_samples_for_this_client)
        self.batches = list(zip(batches_X, batches_y))

    @staticmethod
    def transform_images(img, dataset):
        if torch.is_tensor(img):
            img = img.cpu().detach().numpy()
        img = Image.fromarray(img)
        img = dataset.transform(img)

        return img

    def __iter__(self):
        return self.batches.__iter__()
