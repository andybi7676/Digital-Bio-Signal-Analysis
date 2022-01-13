from pathlib import Path
import os
from zmq import device
from dataset import EmgDataset
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from filter import BandpassFilter1D, NotchFilter1D
from processing import MeanShift1D, Detrend1D, Resample1D, Normalize1D
from segment import SegmentND
from feature_extraction import *
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from model import ResNetClassifier
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter 

# process signal of each channel
def process_signal1d(x, raw_fs=1000, low_fs=1, high_fs=120, notch_fs=60, Q=20, window_size=250, step_size=50, target_fs=512):
    """
    @param x: signal of a single channel
    @param raw_fs: original sampling rate
    @param low_fs: low cutoff frequency
    @param high_fs: high cutoff frequency
    @param notch_fs: notch cutoff frequency
    @param Q: Q factor
    @param window_size: windows size for detrending
    @param step_size: step size for detrending
    @param target_fs: target sampling rate for resampling step
    """
    # mean-correct signal
    x_processed = MeanShift1D.apply(x)
    # filtering noise
    x_processed = BandpassFilter1D.apply(x_processed, low_fs, high_fs, order=4, fs=raw_fs)
    x_processed = NotchFilter1D.apply(x_processed, notch_fs, Q=Q, fs=raw_fs)
    # detrend
    x_processed = Detrend1D.apply(x_processed, detrend_type='locreg', window_size=window_size, step_size=step_size)
    # resample
    x_processed = Resample1D.apply(x_processed, raw_fs, target_fs)
    # rectify
    x_processed = abs(x_processed)
    # normalize
    x_processed = Normalize1D.apply(x_processed, norm_type='min_max')
    return x_processed

# process multi-channel signal
def process_signalnd(x, raw_fs=1000, low_fs=1, high_fs=120, notch_fs=60, Q=20, window_size=250, step_size=50, target_fs=512):
    """
    @param x: signal of a single channel
    @param raw_fs: original sampling rate
    @param low_fs: low cutoff frequency
    @param high_fs: high cutoff frequency
    @param notch_fs: notch cutoff frequency
    @param Q: Q factor
    @param window_size: windows size for detrending
    @param step_size: step size for detrending
    @param target_fs: target sampling rate for resampling step
    """
    num_channels = x.shape[1]
    x_processed = np.array([])
    for i in range(num_channels):
        # process each channel
        channel_processed = process_signal1d(x[:, i], raw_fs, low_fs, high_fs, notch_fs, Q, window_size, step_size, target_fs)
        channel_processed = np.expand_dims(channel_processed, axis=1)
        if i == 0:
            x_processed = channel_processed
            continue
        x_processed = np.hstack((x_processed, channel_processed))
    return x_processed

def get_data():
# parameters for clean data
    raw_fs = 1000
    target_fs = 512
    low_fs = 10
    high_fs = 120
    notch_fs = 60
    Q = 20
    windows = 512
    steps = 50
    # segment parameters
    seg_window_size = 512
    seg_step_size = 32
    # data root
    data_root = './data/classification_data'
    root = Path(data_root)
    data = {}
    label = {}
    for sub in root.iterdir():
        data[sub.name] = [] # create list for `train`,`val`,`test data segments
        label[sub.name] = [] # create list for label of data segments
        for subsub in sub.iterdir():
            if subsub.name == 'Normal':
                y = 0   # set `Normal` class as 0
            elif subsub.name == 'HandOpen':
                y = 1   # set `HandOpen` class as 1
            elif subsub.name == 'HandClose':
                y = 2   # set `HandClose` class as 2
            for filename in subsub.iterdir():
                # load raw signal from file
                emg_raw = pd.read_csv(str(filename)).values
                # clean raw signal
                emg_processed = process_signalnd(emg_raw, raw_fs, low_fs, high_fs, notch_fs, Q, windows, steps, target_fs) # each emg_processed length: (5120, 4)
                # segment signal
                signal_segments = SegmentND.apply(emg_processed, seg_window_size, seg_step_size) # each segment's shape: (512, 4)
                data[sub.name].extend(signal_segments)
                label[sub.name].extend([y] * len(signal_segments))
    print('Number of train samples (X, Y) = ({}, {})'.format(len(data['train']), len(label['train'])))
    print('Number of val samples   (X, Y) = ({}, {})'.format(len(data['val']), len(label['val'])))
    print('Number of test samples  (X, Y) = ({}, {})'.format(len(data['test']), len(label['test'])))
    return data, label

def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""

    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)

    outs = model(data)
    loss = criterion(outs, labels)

    return loss, outs, labels

def test(dataloader, model, criterion, device): 
    """Test on test set."""

    model.eval()
    running_loss = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Test", unit=" uttr")

    accuracy = 0
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, outs, labels = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            preds = outs.argmax(dim=-1).cpu().numpy()
            # print(preds.shape)
            # assert 1==2
            for pred, label in zip(preds, labels):
                if pred == label:
                    accuracy += 1

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
        loss=f"{running_loss / (i+1):.2f}",
        )

    pbar.close()
    model.train()
    accuracy /= len(dataloader)
    print(f"[Info]: We got accuracy {accuracy} in {len(dataloader)} sequences!",flush = True) 

    return running_loss / len(dataloader)

def main():
    out_dir = './DL'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_all, label_all = get_data()
    test_set = EmgDataset(data_all['test'], label_all['test'])

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(
        label_num=3,
        in_channels=[4, 16, 16],
        out_channels=[16, 16, 8],
        downsample_scales=[1, 1, 1],
        kernel_size=5,
        z_channels=8,
        dilation=True,
        leaky_relu=True,
        dropout=0.0,
        stack_kernel_size=3,
        stack_layers=2,
        nin_layers=0,
        stacks=[3, 3, 3],
    ).to(device)
    load_ckpt_file = os.path.join(out_dir, 'resnet_best.ckpt')
    model.load_state_dict(torch.load(load_ckpt_file, map_location=device))
    criterion = nn.CrossEntropyLoss().to(device)

    try:
        testing_loss = test(test_loader, model, criterion, device)
        print(f"testing loss: {testing_loss}")
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f'[Runner] - CUDA out of memory')
            # if self.first_round:
                # raise
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            raise
            # continue
        else:
            raise

if __name__ == "__main__":
    main()
    





