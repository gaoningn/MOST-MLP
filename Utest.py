import os
import torch
import torch.nn as nn
import numpy as np

import models.RoPE-ViT.my_vit_rope as myrope
import models.RLRR-ViT.RLRR as myrlrr
import models.MambaVision.MambaVision as mymamba
import models.MOST-MLP.network as mynetwork

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def calculation_score(cm, n_classes):
    precision_per_class = []
    recall_per_class = []
    specificity_per_class = []
    f1_per_class = []

    for i in range(n_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - TP - FP - FN

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(sensitivity)
        specificity_per_class.append(specificity)
        f1_per_class.append(f1)
    macro_pre = np.mean(precision_per_class)
    macro_sen = np.mean(recall_per_class)
    macro_spe = np.mean(specificity_per_class)
    macro_f1 = np.mean(f1_per_class)
    return macro_pre, macro_sen, macro_spe, macro_f1

def model_test(model, data_path, patient_num, device):
    model.eval()
    correct = 0
    total_samples = 0
    predictions = []
    labels = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for n in range(patient_num):
            test_data = np.load(os.path.join(data_path, f'patient{n}.npz'))
            label = torch.from_numpy(test_data['label'])
            motion_feature = torch.from_numpy(test_data['motion_feature'])
            image_feature = torch.from_numpy(test_data['image_feature'])
            output = model(motion_feature.to(device), image_feature.to(device))
            probs = nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            correct += (predicted == label.to(device)).sum().item()
            total_samples += label.size(0)
            predictions.extend(predicted.cpu().numpy())
            labels.extend(label.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        accuracy = correct / total_samples
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        y_true_flat = label_binarize(all_labels, classes=range(5)).ravel()
        y_prob_flat = all_probs.ravel()
        fpr_micro, tpr_micro, _ = roc_curve(y_true_flat, y_prob_flat)
        micro_auc = auc(fpr_micro, tpr_micro)
        print(f'Acc: {accuracy:.3f}')
        print(f'AUC: {micro_auc:.3f}')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data_path = './data'
    state_dict_path = './parameters/RoPE-ViT.pth'
    model = myrope.My_RopeVit().to(device)
    model.load_state_dict(torch.load(state_dict_path))

    model_test(model, test_data_path, 5, device)
