import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import cross_entropy
from .Utils import *


def test(data_source, model, verbose=True):
    pred_list = []

    data_iterator = data_source
    true_correct = [None] * len(data_iterator)

    model.eval()
    total_loss = 0
    pred_list_single_model = []

    for batch_id, (data, targets) in enumerate(data_iterator):
        output = model(data).detach()
        #targets = targets.type(torch.LongTensor).cuda()
        # sum up batch loss
        total_loss += cross_entropy(output, targets, reduction='sum').item()
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        pred_list.append(pred.cpu().numpy())
        true_correct[batch_id] = targets.data.view_as(pred).cpu().detach()

        del targets, data, output, pred

    model.train()
    pred_list = torch.tensor(np.concatenate(pred_list))
    correct_output = torch.cat(true_correct, 0)
    correct = pred_list.eq(correct_output).cpu().sum().item()

    name = model.name
    dataset_size = pred_list.shape[0]
    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss / dataset_size

    if verbose:
        print_timed(
            f'___Test {name}: Average loss: {total_l:.4f}, Accuracy: {correct}/{dataset_size} ({acc:.4f}%)')

    return acc


def visualize_model_predictions(dataset_to_use, model_to_test, CLASSES, STD_DEV, MEAN, batch_index_to_plot=2, show_labels=True):
    data_batch = list(dataset_to_use)[batch_index_to_plot]
    model_to_test.eval()
    data, labels = data_batch
    output = model_to_test(data).detach()
    pred = output.data.max(1)[1]
    pred = pred.cpu().numpy()
    unique_labels, counts = np.unique(pred, return_counts=True)
    distribution_map = {l: count for l, count in zip(unique_labels, counts)}
    print(f'Prediction Distribution: {distribution_map}')
    fig, axs = plt.subplots(4, 8, dpi=300)
    for x in range(8):
        for y in range(4):
            axs[y, x].imshow(unnormalize_image(
                data[x*8+y].cpu(), STD_DEV=STD_DEV, MEAN=MEAN))
            title = ''
            if show_labels:
                title += f'Label: {CLASSES[labels[x*8+y]]}\n'
            title += f'Prediction: {CLASSES[pred[x*8+y]]}'
            axs[y, x].set_title(title, fontsize=4)
            axs[y, x].get_xaxis().set_visible(False)
            axs[y, x].get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()
    return


def calculate_hash_of_tensor(data):
    """
    Calculates were primitive hash value of a tensor.
    This hash is not secure!
    It is only sufficient to notice accidental changes at the tensors (to notice anoying bugs)
    """
    return float(data.double().mean())


def check_hashs(models, hash_values):
    for model, hashs_of_model in zip(models, hash_values):
        for name, data in model.items():
            hash_value = calculate_hash_of_tensor(data)
            if hash_value != hashs_of_model[name]:
                raise Exception(
                    'Your implementation has changed the benign models.\nThis causes unexpected behavior. Please check you implementation and __retrain all benign models__')
    return


def evaluate_model(model_to_evaluate, test_data, backdoor_test_data, name=None, verbose=True):
    if name is None:
        name = model_to_evaluate.name
    main_task_accuracy = test(test_data, model_to_evaluate, verbose=False)
    backdoor_accuracy = test(
        backdoor_test_data, model_to_evaluate, verbose=False)
    if verbose:
        print_timed(
            f'Performance of {name}: MA={main_task_accuracy:1.2f} BA={backdoor_accuracy:1.2f}')
    return main_task_accuracy, backdoor_accuracy


def model_dist_norm(model1, model2, NAMES_OF_AGGREGATED_PARAMETERS):
    squared_sum = 0
    for name, layer in model1.items():
        if name not in NAMES_OF_AGGREGATED_PARAMETERS:
            continue
        squared_sum += torch.sum(torch.pow(layer.data - model2[name].data, 2))
    return torch.sqrt(squared_sum)


def do_save_division(dividend, divisor, zero_value='-'):
    if divisor == 0:
        return zero_value
    return dividend / divisor


def evaluate_model_filtering(indices_of_accepted_models, number_of_adversaries, number_of_benign_clients):
    indices_of_accepted_models = np.array(indices_of_accepted_models)
    tn = np.where(indices_of_accepted_models <
                  number_of_benign_clients)[0].shape[0]
    assert 0 <= tn <= number_of_benign_clients
    fn = np.where(indices_of_accepted_models >=
                  number_of_benign_clients)[0].shape[0]
    assert 0 <= fn <= number_of_adversaries, f'FN={fn}, number_of_adversaries={number_of_adversaries}, number_of_benign_clients={number_of_benign_clients}, indices_of_accepted_models={indices_of_accepted_models}'
    tp = number_of_adversaries - fn
    assert 0 <= tp <= number_of_adversaries
    fp = number_of_benign_clients - tn
    assert 0 <= fp <= number_of_benign_clients
    tnr = tn/number_of_benign_clients
    assert 0 <= tnr <= 1
    tpr = do_save_division(tp, number_of_adversaries)
    assert 0 <= tpr <= 1
    precision = do_save_division(tp, tp + fp)
    assert 0 <= precision <= 1
    f1_score = do_save_division(2 * tp, 2 * tp + fp + fn)
    assert 0 <= f1_score <= 1
    print_timed(f'TNR = {tnr*100:1.2f}%')
    print_timed(f'TPR = {tp/number_of_adversaries*100:1.2f}% (Recall)')
    print_timed(f'Precision = {precision*100:1.2f}%')
    print_timed(f'F1-Score = {f1_score:1.2f}')


def get_models_hash(all_trained_benign_models):
    hash_values = []

    for model in all_trained_benign_models:
        hashs_of_model = {}
        for name, data in model.items():
            hashs_of_model[name] = calculate_hash_of_tensor(data)
        hash_values.append(hashs_of_model)
    return hash_values
