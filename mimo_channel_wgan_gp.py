"""
Exercise 2.4(b)
Train a GAN-based model (WGAN-GP) on a QuaDRiGa-generated 2x2 MIMO channel dataset.

Expected MATLAB dataset variables:
    H_vec  : [Nsnapshots, 2*Nr*Nt] real/imag interleaved channel vectors
or
    H_flat : [Nr, Nt, Nsnapshots] complex channel matrices
or
    H_mimo : [Nr, Nt, Nsnapshots] complex channel matrices

Example:
    python mimo_channel_wgan_gp.py --data quadriga_mimo_2x2_dataset.mat --epochs 300
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vectorize_complex_channel(H: np.ndarray) -> np.ndarray:
    """
    Convert complex channel tensor [Nr, Nt, Ns] to real-valued features [Ns, 2*Nr*Nt].
    Real/imag are interleaved per coefficient.
    """
    if H.ndim != 3:
        raise ValueError(f"Expected a 3-D tensor [Nr, Nt, Ns], got shape {H.shape}")

    nr, nt, ns = H.shape
    out = np.zeros((ns, 2 * nr * nt), dtype=np.float32)
    for k in range(ns):
        hk = H[:, :, k].reshape(-1, order="F")  # MATLAB column-major convention
        out[k, 0::2] = np.real(hk)
        out[k, 1::2] = np.imag(hk)
    return out


def load_dataset(mat_path: str) -> np.ndarray:
    d = sio.loadmat(mat_path)

    if "H_vec" in d:
        x = d["H_vec"].astype(np.float32)
    elif "H_flat" in d:
        x = vectorize_complex_channel(d["H_flat"])
    elif "H_mimo" in d:
        x = vectorize_complex_channel(d["H_mimo"])
    else:
        raise KeyError(
            "Dataset must contain one of: H_vec, H_flat, H_mimo"
        )

    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got {x.shape}")
    return x


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


# -----------------------------
# Models
# -----------------------------

class Generator(nn.Module):
    def __init__(self, latent_dim: int, data_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, data_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, data_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# WGAN-GP helpers
# -----------------------------

def gradient_penalty(
    critic: Critic,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_samples)

    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates)
    grad_outputs = torch.ones_like(d_interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# -----------------------------
# Training and evaluation
# -----------------------------

def train_wgan_gp(
    x_train: np.ndarray,
    output_dir: str,
    latent_dim: int = 32,
    epochs: int = 300,
    batch_size: int = 256,
    critic_steps: int = 5,
    lr: float = 1e-4,
    gp_lambda: float = 10.0,
) -> tuple[Generator, Standardizer, dict[str, list[float]]]:
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dim = x_train.shape[1]

    standardizer = Standardizer(
        mean=x_train.mean(axis=0, keepdims=True),
        std=x_train.std(axis=0, keepdims=True) + 1e-8,
    )
    x_train_n = standardizer.transform(x_train).astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(x_train_n))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G = Generator(latent_dim, data_dim).to(device)
    D = Critic(data_dim).to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    history = {"critic_loss": [], "generator_loss": []}

    for epoch in range(epochs):
        g_loss_epoch = []
        d_loss_epoch = []

        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            batch_sz = real_batch.size(0)

            for _ in range(critic_steps):
                z = torch.randn(batch_sz, latent_dim, device=device)
                fake_batch = G(z).detach()

                d_real = D(real_batch).mean()
                d_fake = D(fake_batch).mean()
                gp = gradient_penalty(D, real_batch, fake_batch, device)
                d_loss = d_fake - d_real + gp_lambda * gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            z = torch.randn(batch_sz, latent_dim, device=device)
            fake_batch = G(z)
            g_loss = -D(fake_batch).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            d_loss_epoch.append(float(d_loss.item()))
            g_loss_epoch.append(float(g_loss.item()))

        history["critic_loss"].append(np.mean(d_loss_epoch))
        history["generator_loss"].append(np.mean(g_loss_epoch))

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d}/{epochs} | "
                f"Critic loss: {history['critic_loss'][-1]:.4f} | "
                f"Generator loss: {history['generator_loss'][-1]:.4f}"
            )

    torch.save(
        {
            "generator_state_dict": G.state_dict(),
            "latent_dim": latent_dim,
            "data_dim": data_dim,
            "mean": standardizer.mean,
            "std": standardizer.std,
        },
        os.path.join(output_dir, "mimo_wgan_gp.pt"),
    )

    return G, standardizer, history


@torch.no_grad()
def generate_samples(
    G: Generator,
    standardizer: Standardizer,
    n_samples: int,
    latent_dim: int,
    device: torch.device,
) -> np.ndarray:
    G.eval()
    z = torch.randn(n_samples, latent_dim, device=device)
    x_fake_n = G(z).cpu().numpy()
    return standardizer.inverse_transform(x_fake_n).astype(np.float32)


def plot_training_curves(history: dict[str, list[float]], save_path: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history["critic_loss"], label="Critic")
    plt.plot(history["generator_loss"], label="Generator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_real_vs_fake(real_x: np.ndarray, fake_x: np.ndarray, save_path: str) -> None:
    """
    Plot the first complex coefficient H(1,1) as a scatter comparison.
    Since we used interleaved real/imag, coefficient #1 uses columns 0 and 1.
    """
    plt.figure(figsize=(5, 5))
    n = min(len(real_x), len(fake_x), 2000)
    plt.scatter(real_x[:n, 0], real_x[:n, 1], s=8, alpha=0.6, label="Real")
    plt.scatter(fake_x[:n, 0], fake_x[:n, 1], s=8, alpha=0.6, label="Generated")
    plt.xlabel(r"Re$\{H_{11}\}$")
    plt.ylabel(r"Im$\{H_{11}\}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def summarize_statistics(real_x: np.ndarray, fake_x: np.ndarray) -> dict[str, float]:
    real_mean_err = float(np.linalg.norm(real_x.mean(axis=0) - fake_x.mean(axis=0)))
    real_cov = np.cov(real_x.T)
    fake_cov = np.cov(fake_x.T)
    cov_err = float(np.linalg.norm(real_cov - fake_cov, ord="fro"))
    return {
        "mean_error_l2": real_mean_err,
        "covariance_error_fro": cov_err,
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="quadriga_mimo_2x2_dataset.mat")
    parser.add_argument("--out", type=str, default="mimo_gan_results")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    x = load_dataset(args.data)
    print(f"Loaded dataset: {x.shape[0]} snapshots, feature dimension = {x.shape[1]}")

    # train/test split
    n_train = int(0.8 * len(x))
    x_train = x[:n_train]
    x_test = x[n_train:]

    G, standardizer, history = train_wgan_gp(
        x_train=x_train,
        output_dir=args.out,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.to(device)
    x_fake = generate_samples(
        G=G,
        standardizer=standardizer,
        n_samples=len(x_test),
        latent_dim=args.latent_dim,
        device=device,
    )

    # Save generated samples
    sio.savemat(os.path.join(args.out, "generated_mimo_channels.mat"), {"H_vec_fake": x_fake})

    # Save diagnostics
    stats = summarize_statistics(x_test, x_fake)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("WGAN-GP summary\n")
        f.write(f"Test snapshots: {len(x_test)}\n")
        for k, v in stats.items():
            f.write(f"{k}: {v:.6f}\n")

    plot_training_curves(history, os.path.join(args.out, "training_curve.png"))
    plot_real_vs_fake(x_test, x_fake, os.path.join(args.out, "real_vs_fake_H11.png"))

    print("Training complete.")
    print(f"Results saved in: {args.out}")
    print("Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
