import matplotlib.pyplot as plt
import numpy as np
import torch

def poison_single_image(image, label, BACKDOOR_TARGET_CLASS, STD_DEV, MEAN, malignant_client_count: int = None, malignant_client_order_in_distribution = None):
    """Applies a BadNet trigger onto the provided image.
    
    Taken from the week 11 lab notebook (Federated Learning, dr. Picek) and refactored to allow distributed BadNet attacks."""

    triggered_image = image.clone()
    color = (torch.from_numpy(np.array([1, 0, 0])) - MEAN) / STD_DEV
    if malignant_client_order_in_distribution == None:
        triggered_image[:, -6:, -6:] = color.repeat((6, 6, 1)).permute(2, 1, 0)
    else:
        backdoor_height = 6
        # Calculate the width of the backdoor for each individual client's part of the backdoor
        backdoor_width = int(backdoor_height / malignant_client_count)
        
        # Calculate the positions of a client's part of the backdoor so that they can be coordinated.
        client_starting_width = -((malignant_client_order_in_distribution + 1) * backdoor_width)
        client_ending_width = -((malignant_client_order_in_distribution * backdoor_width) + 1) + 1
        
        # If the ending position of a client's part of the backdoor is the end of the image, 
        # then we cannot use the value 0 for slicing, as Python interprets that as the beginning of the image.
        # Instead, we set the value to None so that the slicing happens correctly.
        if client_ending_width == 0:
            client_ending_width = None
        
        # Apply the client's part of the backdoor to the image.
        triggered_image[:, -backdoor_height:, client_starting_width:client_ending_width] = color.repeat((backdoor_height, backdoor_width, 1)).permute(2, 0, 1)
        
    return triggered_image, BACKDOOR_TARGET_CLASS


class BackdoorData:
    """Creates a version of the given data with a backdoor inserted.
    
    Taken from the week 11 lab notebook (Federated Learning, dr. Picek) and refactored to take an Attack object for executing the attack on a single image."""

    def __init__(self, data_loader, attack, pdr, COMPUTATION_DEVICE, BACKDOOR_TARGET_CLASS, STD_DEV, MEAN, malignant_client_order_index=None, is_test_dataloader=False):
        self.batches = []
        self.COMPUTATION_DEVICE = COMPUTATION_DEVICE
        
        if is_test_dataloader:
            attack.distributed = False
        
        for batch in data_loader:
            labels_of_batch = [label.item() for label in batch[1]]
            images_of_batch = [img for img in batch[0]]
            
            poisoning_for_batch = int(len(images_of_batch) * pdr)
            for image_index in range(poisoning_for_batch):
                image = images_of_batch[image_index]
                label = labels_of_batch[image_index]
                
                if attack.distributed:
                    image = attack.execute(image, malignant_client_order_index)
                else:
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