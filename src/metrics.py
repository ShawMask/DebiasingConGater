import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from typing import Union
# from numpy.typing import ArrayLike


def f1score(predictions: torch.Tensor, labels: torch.Tensor, **kwargs): #-> Union[float, ArrayLike[float]]
    pred_np = predictions.numpy()
    labels_np = labels.numpy()
    return f1_score(labels_np, pred_np, **kwargs)


def accuracy(predictions: torch.Tensor, labels: torch.Tensor, balanced: bool = False) -> float:
    pred_np = predictions.numpy()
    labels_np = labels.numpy()
    if balanced:
        return balanced_accuracy_score(labels_np, pred_np)
    return accuracy_score(labels_np, pred_np)

# TODO: GAP metric needs to be added
# def pred_gap(predictions: torch.Tensor, task_labels: torch.Tensor, task_unique:dict,
#              attribute_labels: torch.Tensor, attribute_unique: dict) -> float:
#     TPR = {}
#     # gap = tpr(y=1|M) - tpr(y=1|F) GAP = sqrt(sum(gap**2)/label)
#     for  in attibute_labels[target_att].keys():
#         df[f"TP_{j}"] = []
#         df[f"acc_{j}"] = []
#         for k in task_labels.keys():
#             df[f"TP_{j}_{k}"] = []
#             df[f"TPR_{j}_{k}"] = []
#             df[f"acc_{j}_{k}"] = []
#             df[f"gap_{k}"] = []
#
#     for task_label, task_value in task_labels_unique.items():
#         df = {"acc": [], "gap": [], "GAP": []}
#         for k, v in attribute_unique.items():
#             mask = attribute_label == v
#             TPR[k] = (predictions == labels)[mask]/torch.sum(mask)
#     GAP =
#
#     return accuracy_score(labels_np, pred_np)
