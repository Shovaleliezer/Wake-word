
import torch
from torch import nn
from  torchaudio.datasets import SPEECHCOMMANDS
from torch import optim
import torch.nn.functional as F
import librosa
import librosa.display
from torch.optim import lr_scheduler
import os
import IPython.display as ipd
import torchaudio
from tqdm.notebook import tqdm

class KeywordSpotting(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.sigmoid(x)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class Subset(SPEECHCOMMANDS):
    def __init__(self,subset, str=None):
        super().__init__('./', download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileob:
                return [os.path.join(self._path, line.strip()) for line in fileob]
        if subset== "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

def label_to_index(word,key_word='marvin'):
    # Return the position of the word in labels
    if word==key_word:
        return torch.tensor(1.0)
    else:
        return torch.tensor(0.0)


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]



def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.binary_cross_entropy(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # record loss
        losses.append(loss.item())
    return model
def number_of_correct(pred, target):
    # count number of correct predictions
    acc = pred.squeeze().eq(target).sum().item()
    true_pos = 0
    false_pos = 0
    pos = 0
    for i, t in enumerate(target):
        if t==1:
            pos +=1
            if t==pred[i]:
                true_pos +=1
        else:
            if t!=pred[i]:
                false_pos +=1

    return acc, true_pos, false_pos, pos


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return torch.round(tensor)

def predict(tensor):
    # Use the model to predict the label of the waveform
    tensor = tensor.to(device)
    tensor = transform(tensor)
    tensor = tensor.reshape(1,1,-1)
    tensor = model(tensor)
    tensor = get_likely_index(tensor)
    return tensor

def test(model, epoch):
    model.eval()
    correct = 0
    true_correct = 0
    positive = 0
    false_correct = 0
    recall = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        acc, true_pos, false_pos, pos = number_of_correct(pred, target)
        correct += acc
        true_correct += true_pos
        false_correct += false_pos
        positive += pos
    recall = true_correct / positive
    prec =  true_correct / (true_correct+false_correct)
    print(f"\nTest Epoch: {epoch}\trecall: {recall}\n")
    print(f"\nTest Epoch: {epoch}\tprec: {prec}\n")
    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set = Subset("training")
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

    test_set = Subset('testing')
    print('1')
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    testL = sorted(list(set(datapoint[2] for datapoint in test_set)))
    print('s')
    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    transformed = transform(waveform)
    ipd.Audio(transformed.numpy(), rate=new_sample_rate)
    batch_size = 256

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


    model = KeywordSpotting(n_input=transformed.shape[0], n_output=len(labels))
    model.to(device)
    print(model)

    n = count_parameters(model)
    print("Number of parameters: %s" % n)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    log_interval = 20
    n_epoch = 25

    pbar_update = 1 / (len(train_loader) + len(test_loader))
    losses = []

    # The transform needs to live on the same device as the model and the data.
    transform = transform.to(device)

    for epoch in range(1, n_epoch + 1):
        model = train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()

    waveform, sample_rate, utterance, *_ = train_set[7178]
    torch.save(model.state_dict(), 'model6.pth')
    torch.save(model, 'model7.pth')
    print(f"Expected: {label_to_index(utterance)}. Predicted: {predict(waveform)}.")
    for i in range(7178, 7310):
        waveform, sample_rate, label, speaker_id, utterance_number = test_set[i]
        waveform1, sample_rate1, label1, speaker_id1, utterance_number1 = test_set[0]
        shape = waveform1.shape
        if waveform.shape==shape:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
            waveform = waveform.reshape(1, 1, -1)
            pred = predict(waveform)
            print(f"epoch: {i}, Expected: {label_to_index(label)}. Predicted: {predict(waveform).squeeze()}.")
        else:
            print(waveform.shape)
    print(model.state_dict())