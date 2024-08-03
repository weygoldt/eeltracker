import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from xgboost import XGBRegressor
import matplotlib.pyplot as plt

neels = 30
colors = sns.color_palette("tab10", neels)
print(len(colors))


def main():
    # Load the data
    data = np.load("simulations.npz")
    print(data.files)

    # Target positions (5 nodes, 3D coordinates)
    xs = data["x"]  # Shape: (n_samples, 5)
    ys = data["y"]  # Shape: (n_samples, 5)
    zs = data["z"]  # Shape: (n_samples, 5)

    # Plot random subset of data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(neels):
        rand_idx = np.random.randint(0, xs.shape[0])
        ax.scatter(xs[rand_idx], ys[rand_idx], zs[rand_idx], alpha=0.6)
        ax.plot(xs[rand_idx], ys[rand_idx], zs[rand_idx], alpha=0.6)
    plt.show()

    # Take only the first point
    # xs = xs[:, 0]  # Shape: (n_samples,)
    # ys = ys[:, 0]  # Shape: (n_samples,)
    # zs = zs[:, 0]  # Shape: (n_samples

    # Features
    amps = data["amps"]  # Shape: (n_samples, 16)
    amps = amps.reshape(amps.shape[0], -1)  # Shape: (n_samples, 16)

    # norm amps to max for each sample
    amps = amps / np.max(amps, axis=1)[:, None]

    # invert amps to relate to distance from 0
    # amps = 1 / np.sqrt(amps)

    signs = data["signs"]  # Shape: (n_samples, 16)
    signs = signs.reshape(signs.shape[0], -1)  # Shape: (n_samples, 16)

    amps = amps * signs  # Combine amplitude and sign

    # Combine amplitude and sign as input features
    # inputs = np.concatenate([amps, signs], axis=1)  # Shape: (n_samples, 32)
    inputs = amps
    print(f"Inputs shape: {inputs.shape}")

    # Stack x, y, z coordinates as the target output
    outputs = np.stack([xs, ys, zs], axis=-1)  # Shape: (n_samples, 5, 3)
    outputs = outputs.reshape(outputs.shape[0], -1)  # Shape: (n_samples, 15)
    print(f"Outputs shape: {outputs.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, outputs, test_size=0.2, random_state=42
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the MLP regressor
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 528, 256, 128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=8000,
        random_state=8,
        learning_rate_init=0.001,
        learning_rate="adaptive",
    )

    # Train the model
    mlp.fit(X_train, y_train)

    # Evaluate the model (optional)
    score = mlp.score(X_test, y_test)
    print(f"Test set R^2 score: {score:.4f}")

    # Predict and visualize (optional)
    y_pred = mlp.predict(X_test)
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
        # make connecting lines
        # x = np.array([target_nodes[0][0], pred_nodes[0][0]])
        # y = np.array([target_nodes[0][1], pred_nodes[0][1]])
        # z = np.array([target_nodes[0][2], pred_nodes[0][2]])
        # ax.plot(x, y, z, color="k", alpha=0.6)
    plt.legend()
    plt.show()

    # Plot error as function of distance from 0
    distances = np.linalg.norm(y_test, axis=1)
    errors = np.linalg.norm(y_test - y_pred, axis=1)
    plt.scatter(distances, errors, alpha=0.6)
    plt.xlabel("Distance from origin")
    plt.ylabel("Error")
    plt.show()
    """
    # Define the XGBoost regressor for each coordinate output
    models = [
        XGBRegressor(
            n_estimators=1000, max_depth=30, learning_rate=00.1, random_state=32
        )
        for _ in range(outputs.shape[1])
    ]

    # Train each model separately
    for i, model in enumerate(models):
        model.fit(X_train, y_train[:, i])

    # Evaluate the model (optional)
    y_pred = np.column_stack([model.predict(X_test) for model in models])
    score = [model.score(X_test, y_test[:, i]) for i, model in enumerate(models)]
    print(f"Test set R^2 scores: {score}")

    # Predict and visualize (optional)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for target, pred in zip(y_test[:15], y_pred[:15]):
        target_nodes = target.reshape(5, 3)
        pred_nodes = pred.reshape(5, 3)
        # target_nodes = target
        # pred_nodes = pred
        ax.scatter(*target_nodes.T, color="blue", label="True", alpha=0.6)
        ax.scatter(*pred_nodes.T, color="red", label="Predicted", alpha=0.6)
        ax.plot(
            target_nodes[:, 0],
            target_nodes[:, 1],
            target_nodes[:, 2],
            color="blue",
            alpha=0.6,
        )
        ax.plot(
            pred_nodes[:, 0], pred_nodes[:, 1], pred_nodes[:, 2], color="red", alpha=0.6
        )
    plt.legend()
    plt.show()
    """


if __name__ == "__main__":
    main()
