from IPython.display import display
import wfdb
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Separating patients in 2 sets (training, test sets), as in Chazal et al. 2004
training_patients = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
testing_patients = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

# Equivalence between MIT-BIH and AAMI
dic = {'N' : 'N', '.' : 'N', 'L' : 'N', 'R' : 'N', 'e' : 'N', 'j' : 'N',
       'A' : 'S', 'a' : 'S', 'J' : 'S', 'S' : 'S',
       'V' : 'V', 'E' : 'V',
       'F' : 'F'}

class_idx = {'N' : 0, 'S' : 1, 'V' : 2, 'F' : 3} 

# Function to extract beats and label for one record
def extract_beats(record_nb, before=90, after=110):
    record_name = '../../mitbih/' + record_nb
    rec = wfdb.rdrecord(record_name)
    ann = wfdb.rdann(record_name, 'atr') #ann.sample has the beat peak indexes, ann.symbol has the corresponding MIT-BIH beat type
    channel = np.where(np.array(rec.sig_name) == "MLII")
    sig = rec.p_signal[:, channel[0]]

    beats, types = [],[]
    for peak, symbol in zip(ann.sample, ann.symbol):
        if symbol not in dic :
            continue
        if peak - before < 0 or peak + after > len(sig) : 
            continue
        beat = sig[peak-before : peak+after]
        type = class_idx[dic[symbol]]
        beats.append(beat)
        types.append(type)

    return np.array(beats), np.array(types)

# Creating the datasets
def build_datasets(record_nbs): 
    X, y = [],[]
    for record_nb in record_nbs : 
        beats, types = extract_beats(record_nb)
        X.append(beats)
        y.append(types)
    return np.concatenate(X), np.concatenate(y)

X_train, y_train = build_datasets(training_patients)
X_test, y_test = build_datasets(testing_patients)

print(f"Training sets shape  : {X_train.shape, y_train.shape}")
print(f"Testing sets shape : {X_test.shape, y_test.shape}")


# CNN expects : (batch, channel, length) but we have (nb_beats, window_size, 1) where window_size is the length of the signal, and 1 is the single channel, 
# To do : transform sets into tensors, transform the dimensions, and creates batches (Dataloader)
# We do it with both : unmodified training set (adding class weight after), and resample training set (no adding class weight after)

# 1. Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0,2,1)
y_train_t = torch.tensor(y_train, dtype=torch.long)

X_test_t = torch.tensor(X_test, dtype=torch.float32).permute(0,2,1)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# 2. Datasets
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

# 3. Dataloader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print("Dataloader created")

# Network
# When entering the network, a sample is of size  : (batch, channel=1, length=window_size)
class BeatCNN(nn.Sequential):
    def __init__(self, n_classes=4):
        super().__init__(
            # Conv and pooling layers
            nn.Conv1d(1, 16, kernel_size=5),
            nn.BatchNorm1d(16), # Normalize over the batch (mean 0, var 1)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1), # Averages each cannel over the length (batch, channel=64, length=window_size) -> (batch, channel=64, 1)

            nn.Flatten(), #(batch, 64, 1) -> (batch, 64)

            # Linear layers
            nn.Linear(64, n_classes),
        )

# Training fct 
def train(num_epochs, train_loader, test_loader, network, criterion, optimizer, device, target_names = ["Normal", "S-beat", "V-beat", "F-beat"]):

    for i in range(num_epochs):
        print(f"Epoch number {i+1} starting")

        network.train()
        for x, y in train_loader : 
            x, y = x.to(device), y.to(device)
            y_pred = network(x)

            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        # Eval once per epoch
        network.eval()
        pred_classes = []
        true_classes = []
        with torch.no_grad():
            for x, y in test_loader :
                x, y = x.to(device), y.to(device)
                y_pred = network(x) #network outputs proba per class
                y_pred_class = y_pred.argmax(1)  # class is the most probable one
                pred_classes.append(y_pred_class)
                true_classes.append(y)

        pred_classes = torch.cat(pred_classes).cpu().numpy()
        true_classes = torch.cat(true_classes).cpu().numpy()
        print(classification_report(y_true=true_classes, y_pred=pred_classes, target_names=target_names))
        cm = confusion_matrix(y_true=true_classes, y_pred=pred_classes)
        """
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[f"{class_name}" for class_name in target_names],
                    yticklabels=[f"{class_name}" for class_name in target_names]
                )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.show()
        """


# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
device = torch.device(device)

# Net init
net = BeatCNN(n_classes=4)
net.to(device)

# Weighted classes
w = compute_class_weight("balanced", classes=np.array([0,1,2,3]), y=y_train)
weights = torch.tensor(w, dtype=torch.float32).to(device)
criterion= nn.CrossEntropyLoss(weight=weights) 

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

train(20, train_loader, test_loader, net, criterion, optimizer, device)