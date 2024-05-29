import numpy as np
import torch

def poison_single_image(image, label, BACKDOOR_TARGET_CLASS, STD_DEV, MEAN):
    triggered_image = image.clone()
    color = (torch.from_numpy(np.array([1, 0, 0])) - MEAN) / STD_DEV
    triggered_image[:, -5 :, -5:] = color.repeat((5, 5, 1)).permute(2, 1, 0)
    return triggered_image, BACKDOOR_TARGET_CLASS


class BackdoorData:
    """Creates a version of the given data with a backdoor inserted.
    
    Taken from the week 11 lab notebook (Federated Learning, dr. Picek) and refactored to take an Attack object for executing the attack on a single image."""

    def __init__(self, data_loader, attack, pdr, COMPUTATION_DEVICE, BACKDOOR_TARGET_CLASS, STD_DEV, MEAN, is_test_dataloader=False):
        self.batches = []
        self.COMPUTATION_DEVICE = COMPUTATION_DEVICE
        for batch in data_loader:
            labels_of_batch = [label.item() for label in batch[1]]
            images_of_batch = [img for img in batch[0]]
            
            poisoning_for_batch = int(len(images_of_batch) * pdr)
            for image_index in range(poisoning_for_batch):
                image = images_of_batch[image_index]
                label = labels_of_batch[image_index]
                # image, label = poison_single_image(image, label, BACKDOOR_TARGET_CLASS=BACKDOOR_TARGET_CLASS, STD_DEV=STD_DEV, MEAN=MEAN)
                image = attack.execute(image)
                
                if is_test_dataloader:
                    labels_of_batch[image_index] = label
                else:
                    labels_of_batch[image_index] = BACKDOOR_TARGET_CLASS
                
                images_of_batch[image_index] = image
            labels_of_batch = torch.from_numpy(np.array(labels_of_batch))
            labels_of_batch = labels_of_batch.to(dtype=torch.int64)
            images_of_batch = torch.stack(images_of_batch)
            self.batches.append((images_of_batch, labels_of_batch))

    def __iter__(self):
        return self.batches.__iter__()
    
    def __len__(self):
        return len(self.batches)
    
    def cuda(self):
        self.batches = [(images.to(self.COMPUTATION_DEVICE), labels.to(self.COMPUTATION_DEVICE)) for images, labels in self.batches]
        
    def cpu(self):
        self.batches = [(images.cpu(), labels.cpu()) for images, labels in self.batches]