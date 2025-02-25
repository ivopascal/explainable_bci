import pickle as pkl

import numpy as np
import shap
import torch
import torchmetrics
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch import nn
from torch.utils.data import DataLoader, Dataset
import mne
from torcheeg.models import EEGNet
import pytorch_lightning as pl


class EEGDataset(Dataset):
    def __init__(self, dataset_from_pkl):
        self.x = [sample[0] for sample in dataset_from_pkl]
        self.y = [sample[1][4] for sample in dataset_from_pkl]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], (self.y[idx] - 1) * -1


class DataModule(pl.LightningDataModule):
    def __init__(self, train_set, val_set, test_set, batch_size, test_batch_size=None):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size if test_batch_size else 1

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False)


class EEGNetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        sampling_rate = 512
        self.eegnet = nn.Sequential(
            EEGNet(chunk_size=307,
                   F1=8,
                   F2=16,
                   D=2,
                   num_electrodes=64,
                   num_classes=1,
                   kernel_1=int(sampling_rate / 2),
                   kernel_2=int(sampling_rate / 8)))

        self.sigmoid = nn.Sigmoid()
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
        shape = x.shape
        x = x.view(shape[0], 1, shape[1], shape[2])

        out = self.eegnet(x)
        out = self.sigmoid(out)
        return out

    def loss_fn(self, out, target):
        return nn.BCELoss()(out, target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     betas=(0.9, 0.999),
                                     weight_decay=0)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        out = self(x)
        loss = self.loss_fn(out[:, 0], y.type_as(out))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        out = self(x)
        loss = self.loss_fn(out[:, 0], y.type_as(out))
        accu = self.accuracy(out[:, 0] > 0.5, y)
        self.log('val_loss', loss)
        self.log('val_acc', accu)
        return loss, accu


def plot_time_dimension(raw, shap, channel_names, ax, fig):
    raw_channel = raw[channel_names.index("FCz"), :].numpy()
    color_data = shap.mean(axis=0).squeeze()
    x = np.linspace(0, 600, 307)

    points = np.array([x, raw_channel]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    climit = max(abs(color_data.min()), abs(color_data).max())
    norm = plt.Normalize(-climit, climit)
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    # Set the values used for colormapping
    lc.set_array(color_data)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)

    # Use a boundary norm instead
    cmap = ListedColormap(['r', 'g', 'b'])
    norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(color_data)
    lc.set_linewidth(2)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-20, 20)


def main():
    dataset_path = "datasets/uncorrected_dataset.pkl"
    with open(dataset_path, "rb") as f:
        train_set, val_set, test_set = pkl.load(f)

    model = EEGNetModel()
    dm = DataModule(EEGDataset(train_set), EEGDataset(val_set), EEGDataset(test_set), batch_size=128)

    comet_logger = pl.loggers.CometLogger(
        api_key="3xX4JIrZCsKBpMeFSsbQBfh0W",
        project_name="EEG-XAI",
        workspace="ivopascal",
        experiment_name="Development",
        save_dir='logs/',
    )

    max_epochs = 300
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=comet_logger,
        accelerator="mps",
        devices=1,
        precision=32,
        log_every_n_steps=30,
    )

    trainer.fit(model, datamodule=dm)
    comet_logger.experiment.end()

    background = torch.cat([entry[0] for entry in dm.val_dataloader()])
    e = shap.DeepExplainer(model, background)

    test_batch = [sample[0] for sample in dm.test_dataloader()]

    test_batch_y = torch.stack([sample[1] for sample in dm.test_dataloader()]).squeeze()
    eeg_batch = torch.stack(test_batch).squeeze()
    sample = torch.randperm(eeg_batch.shape[0])[:200]
    test_batch_y = test_batch_y[sample]
    eeg_batch = eeg_batch[sample, :, :]
    shap_values = e.shap_values(eeg_batch)
    y_pred = model.forward(eeg_batch).squeeze()

    channel_names = ["Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1", "C3", "C5",
                     "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7", "PO3", "O1", "Iz", "Oz",
                     "POz",
                     "Pz", "CPz", "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
                     "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8",
                     "P10",
                     "PO8", "PO4", "O2"]

    info = mne.create_info(channel_names, sfreq=512, ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'))

    true_positives = ((test_batch_y == 1) & (y_pred > 0.5))
    true_negatives = ((test_batch_y == 0) & (y_pred < 0.5))
    false_positives = ((test_batch_y == 0) & (y_pred > 0.5))
    false_negatives = ((test_batch_y == 1) & (y_pred < 0.5))

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(f"Spatial contributions of EEG to ErrP prediction. {max_epochs} epochs")
    axes[0, 0].set_title("True Positives")
    mne.viz.plot_topomap(shap_values[true_positives, :, :].mean(axis=0).mean(axis=1), pos=info, ch_type='eeg', axes=axes[0, 0])

    axes[0, 1].set_title("False Positives")
    mne.viz.plot_topomap(shap_values[false_positives, :, :].mean(axis=0).mean(axis=1), pos=info, ch_type='eeg', axes=axes[0, 1])

    axes[1, 0].set_title("False Negatives")
    mne.viz.plot_topomap(shap_values[false_negatives, :, :].mean(axis=0).mean(axis=1), pos=info, ch_type='eeg', axes=axes[1, 0])

    axes[1, 1].set_title("True Negatives")
    mne.viz.plot_topomap(shap_values[true_negatives, :, :].mean(axis=0).mean(axis=1), pos=info, ch_type='eeg', axes=axes[1, 1])
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(f"Temporal contributions of EEG to ErrP prediction. {max_epochs} epochs")

    plot_time_dimension(eeg_batch[true_positives, :, :].mean(axis=0), shap_values[true_positives, :, :].mean(axis=0), channel_names, ax=axes[0, 0], fig=fig)
    axes[0, 0].set_title("True Positives")
    plot_time_dimension(eeg_batch[false_positives, :, :].mean(axis=0), shap_values[false_positives, :, :].mean(axis=0), channel_names, ax=axes[0, 1], fig=fig)
    axes[0, 1].set_title("False Positives")
    plot_time_dimension(eeg_batch[false_negatives, :, :].mean(axis=0), shap_values[false_negatives, :, :].mean(axis=0), channel_names, ax=axes[1, 0], fig=fig)
    axes[1, 0].set_title("False Negatives")
    plot_time_dimension(eeg_batch[true_negatives, :, :].mean(axis=0), shap_values[true_negatives, :, :].mean(axis=0), channel_names, ax=axes[1, 1], fig=fig)
    axes[1, 1].set_title("True Negatives")
    plt.show()


if __name__ == "__main__":
    main()
