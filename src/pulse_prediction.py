import numpy as np
from pathlib import Path
from trainnn import EelPoseModel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch

model = EelPoseModel.load_from_checkpoint(
    "../models/epose-epoch=149-val_loss=367.05.ckpt",
    input_dim=16,
    output_dim=15,
)
model.eval()


def main():
    files = list(Path("../data/processed/").glob("*.npz"))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for file in files:
        print(file)
        data = np.load(file)
        print(data.files)

        snippets = data["all_snippets"]
        if len(snippets) == 0:
            continue
        snippet_amps = data["all_snippet_amps"]
        snippet_signs = data["all_snippet_signs"]

        amps = snippet_amps / np.max(snippet_amps, axis=1)[:, None]
        amps = amps * snippet_signs

        scaler = StandardScaler()
        amps = scaler.fit_transform(amps)

        lengths = []
        for i in range(len(snippets)):
            snippet = snippets[i]
            amp = amps[i]
            amp = torch.Tensor(amp).to(model.device)

            y_hat = model(amp)
            y_hat = y_hat.cpu().detach().numpy()
            y_hat = y_hat.reshape(5, 3)

            ax.plot(y_hat[:, 0], y_hat[:, 1], y_hat[:, 2], alpha=0.05, c="k")

            length = np.linalg.norm(y_hat[0] - y_hat[-1])
            lengths.append(length)

    print(f"Mean length: {np.mean(lengths)}+-{np.std(lengths)}")

    ax.set_xlim(-400, 400)
    ax.set_ylim(-400, 400)
    ax.set_zlim(-120, 120)
    ax.set_aspect("equal")

    # top down view
    ax.view_init(elev=90, azim=0)
    plt.show()

    fig, ax = plt.subplots()
    ax.hist(lengths, bins=50)
    plt.show()


if __name__ == "__main__":
    main()
