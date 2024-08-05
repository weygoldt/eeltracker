from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pathlib import Path
import shutil
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything, Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import LightningModule


logdir = Path(__file__).parent / "./tensorboard_logs"

try:
    shutil.rmtree(logdir)
except FileNotFoundError:
    pass
logdir.mkdir(exist_ok=True)

neels = 30
colors = sns.color_palette("tab10", neels)

seed_everything(42)


class EelPoseNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EelPoseNet, self).__init__()
        activation = nn.LeakyReLU
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            activation(),
            nn.Linear(256, 528),
            activation(),
            nn.Linear(528, 528),
            activation(),
            nn.Linear(528, 528),
            activation(),
            nn.Linear(528, 528),
            activation(),
            nn.Linear(528, 256),
            activation(),
            nn.Linear(256, 128),
            activation(),
            nn.Linear(128, 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class EelPoseModel(LightningModule):
    def __init__(self, input_dim, output_dim, learning_rate=0.001):
        super(EelPoseModel, self).__init__()
        self.model = EelPoseNet(input_dim, output_dim)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        # self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.5,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }


def load_data():
    data = np.load("simulations.npz")

    # x,y,z coordinates of simulated nodes
    xs = data["x"]
    ys = data["y"]
    zs = data["z"]
    outputs = np.stack([xs, ys, zs], axis=-1)
    outputs = outputs.reshape(outputs.shape[0], -1)

    # sign of the simulated electric field for each electrode
    signs = data["signs"]
    signs = signs.reshape(signs.shape[0], -1)

    # amplitude of the simulated electric field for each electrode
    amps = data["amps"]
    amps = amps.reshape(amps.shape[0], -1)
    amps = amps / np.max(amps, axis=1)[:, None]

    # combine amplitude and sign as input features
    amps = amps * signs
    inputs = amps

    return inputs, outputs


def split_and_scale(inputs, outputs):
    x_train, x_test, y_train, y_test = train_test_split(
        inputs,
        outputs,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


def main():
    inputs, outputs = load_data()
    x_train, x_test, y_train, y_test = split_and_scale(inputs, outputs)

    train_ds = torch.utils.data.TensorDataset(
        torch.Tensor(x_train),
        torch.Tensor(y_train),
    )

    test_ds = torch.utils.data.TensorDataset(
        torch.Tensor(x_test),
        torch.Tensor(y_test),
    )

    batch_size = 56

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=11,
    )

    val_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=11,
    )

    hparams = {
        "input_dim": x_train.shape[1],
        "output_dim": y_train.shape[1],
    }

    model = EelPoseModel(
        input_dim=hparams["input_dim"],
        output_dim=hparams["output_dim"],
        learning_rate=0.001,
    )

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=4,
        monitor="val_loss",
        mode="min",
        dirpath="../models",
        filename="epose-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer(
        max_epochs=150,
        logger=TensorBoardLogger(logdir, name="fabric_logs"),
        deterministic=True,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model,
        train_loader,
        val_loader,
    )

    # Evaluate the model
    y_pred = model(torch.Tensor(x_test)).detach().numpy()

    # Evaluate R2 score
    r2 = r2_score(y_test, y_pred)
    print(f"R2 score: {r2}")

    # Evaluate RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse}")

    # Evaluate MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse}")

    # Evaluate MAE
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"MAE: {mae}")

    # Predict and visualize (optional)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # as many colors as neels
    for i, (target, pred) in enumerate(zip(y_test[:neels], y_pred[:neels])):
        target_nodes = target.reshape(5, 3)
        pred_nodes = pred.reshape(5, 3)
        # target_nodes = target.reshape(1, 3)
        # pred_nodes = pred.reshape(1, 3)
        ax.scatter(*target_nodes.T, color=colors[i], label="True", alpha=0.3)
        ax.scatter(*pred_nodes.T, color=colors[i], label="Predicted", alpha=0.8)
        ax.plot(
            target_nodes[:, 0],
            target_nodes[:, 1],
            target_nodes[:, 2],
            color=colors[i],
            alpha=0.3,
        )
        ax.plot(
            pred_nodes[:, 0],
            pred_nodes[:, 1],
            pred_nodes[:, 2],
            color=colors[i],
            alpha=0.8,
        )
    plt.legend()
    plt.show()

    # Plot error as function of distance from 0
    distances = np.linalg.norm(y_test, axis=1)
    errors = np.linalg.norm(y_test - y_pred, axis=1)
    plt.scatter(distances, errors, alpha=0.6)
    plt.xlabel("Distance from origin")
    plt.ylabel("Error")
    plt.show()

    # Plot error as function of orientation from 0
    orientations = np.arctan2(y_test[:, 1], y_test[:, 0])
    errors = np.linalg.norm(y_test - y_pred, axis=1)
    plt.scatter(orientations, errors, alpha=0.6)
    plt.xlabel("Orientation")
    plt.ylabel("Error")
    plt.show()


if __name__ == "__main__":
    main()
