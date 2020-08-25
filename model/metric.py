import torch
import numpy as np


def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():

        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def mean_points_error(output, target):
    with torch.no_grad():
        if not isinstance(output, np.ndarray):
            output = output.data.cpu().numpy()
            target = target.data.cpu().numpy()
        output_reshape = output.reshape((-1, 3))
        target_reshape = target.reshape((-1, 3))
        error = np.mean(np.sqrt(np.sum(np.square((output_reshape - target_reshape)), axis=1)))
    return error


def mean_points_error_sequence(output, target):
    with torch.no_grad():
        if not isinstance(output, np.ndarray):
            output = output.data.cpu().numpy()
            target = target.data.cpu().numpy()
        output_reshape = output.reshape((-1, 3))
        target_reshape = target.reshape((-1, 3))
        # if len(output.shape) == 3:
        #     # output_reshape = output.reshape((output.shape[0] * int(output.shape[1]/2)*output.shape[2], 2))
        #     output_reshape = output.reshape((-1, 3))
        #     # target_reshape = target.reshape((output.shape[0] * int(output.shape[1]/2)*output.shape[2], 2))
        #     target_reshape = target.reshape((-1, 3))
        # elif len(output.shape) == 2:
        #     # output_reshape = output.reshape((output.shape[0] * int(output.shape[1] / 2), 2))
        #     # target_reshape = target.reshape((output.shape[0] * int(output.shape[1] / 2), 2))
        error = np.mean(np.sqrt(np.sum(np.square((output_reshape - target_reshape)), axis=1)))
    return error