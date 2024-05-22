import numpy as np
from tqdm import tqdm
import torch
import datetime
from collections import defaultdict


def format_for_logging(text):
    text = str(text).split('\n')
    current_time = str(datetime.datetime.now())
    text = ['{0}: {1}'.format(current_time, line) for line in text]
    text = '\n'.join(text)
    return text


def print_timed(text):
    text = format_for_logging(text)
    print(text)


def sort_samples_by_labels(dataset):
    training_data_by_labels = defaultdict(list)
    all_training_images = []
    for ind, (_, label) in tqdm(enumerate(dataset)):
        training_data_by_labels[label].append(ind)
        all_training_images.append(ind)
    all_labels = sorted(list(training_data_by_labels.keys()))
    training_data_by_labels = {label: np.array(images)
                     for label, images in training_data_by_labels.items()}
    return training_data_by_labels, all_labels, np.array(all_training_images)


def create_client_distributions(total_client_number, iid_rate, samples_per_client, all_labels, train_data_by_labels, all_training_images):
    clients_main_labels = np.random.choice(
        all_labels, size=total_client_number)
    samples_of_main_class_per_client = int((1 - iid_rate) * samples_per_client)
    samples_of_all_classes_per_client = samples_per_client - \
        samples_of_main_class_per_client
    print_timed(
        f'Samples from main class per client: {samples_of_main_class_per_client}')
    print_timed(
        f'Samples from all classes per client: {samples_of_all_classes_per_client}')
    indices_for_clients = []
    main_labels_dict = {}
    for client_index, main_label in enumerate(clients_main_labels):
        indices_of_current_client = -1 * np.ones(samples_per_client)
        indices_for_main_label = np.random.choice(indices_of_current_client.shape[0],
                                                  samples_of_main_class_per_client, replace=False)
        assert indices_for_main_label.shape[0] == samples_of_main_class_per_client
        indices_of_current_client[indices_for_main_label] = -2
        indices_for_other_labels = np.where(indices_of_current_client == -1)[0]
        indices_of_current_client[indices_for_main_label] = np.random.choice(
            train_data_by_labels[main_label], samples_of_main_class_per_client, replace=False)
        other_images = np.random.choice(
            all_training_images, samples_of_all_classes_per_client, replace=False)
        indices_of_current_client[indices_for_other_labels] = other_images
        indices_for_clients.append(indices_of_current_client.astype(int))
        main_labels_dict[client_index] = main_label
    print_timed(f'Main label for clients: {main_labels_dict} ')
    return indices_for_clients, main_labels_dict


def batchify(source, batch_size, total_number):
    source = [x for x in source]
    result = []
    for batch_start in range(0, total_number, batch_size):
        batch_end = min(total_number, batch_start + batch_size)
        to_add = source[batch_start:batch_end]
        assert len(to_add) > 0
        if isinstance(to_add[0], tuple):
            tensor = torch.stack([p[0] for p in to_add])
            label = torch.LongTensor([p[1] for p in to_add])
            to_add = [tensor, label]
        else:
            to_add = torch.stack(to_add)
        result.append(to_add)
    return result


def unnormalize_image(tensor, STD_DEV, MEAN):
    result = tensor.clone()
    for t, m, s in zip(result, MEAN, STD_DEV):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    if result.min() < 0:
        assert result.min() > - (10 ** -6)
        result -= result.min()
    return result.transpose(0, 1).transpose(1, 2)
