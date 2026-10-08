"""A small, complete variational autoencoder (VAE) for binarized MNIST.

Companion code for the YouTube lecture "Variational Autoencoders, Visually Explained".

Install: python -m pip install -r requirements.txt
Run:     python vae.py --epochs 10 --latent-dim 2 --out outputs

Images are binarized, so the Bernoulli likelihood (binary cross-entropy) is a literal model choice.
Every loss is reported in nats per image: pixel and latent terms are summed per example, then averaged.
"""
import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class VAE(nn.Module):
    """Encoder q_phi(z|x) = N(mu, diag(sigma^2)); decoder p_theta(x|z) = Bernoulli(sigmoid(logits))."""

    def __init__(self, latent_dim=2, hidden=400):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, 784))

    def encode(self, x):
        h = self.enc(x.flatten(1))
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        # z = mu + sigma * eps keeps the randomness off the gradient path
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        logits = self.decode(self.reparameterize(mu, logvar))
        return logits, mu, logvar


def loss_terms(logits, x, mu, logvar):
    """Negative ELBO = reconstruction (BCE summed over pixels) + closed-form KL to N(0, I)."""
    recon = F.binary_cross_entropy_with_logits(logits, x.flatten(1), reduction="none").sum(1)
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(1)
    return (recon + kl).mean(), recon.mean(), kl.mean()


def binarize(x):
    return (x >= 0.5).float()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, count = torch.zeros(3, device=device), 0
    for x, _ in loader:
        x = x.to(device)
        logits, mu, logvar = model(x)
        total += torch.stack(loss_terms(logits, x, mu, logvar)) * x.shape[0]
        count += x.shape[0]
    return (total / count).cpu().tolist()


@torch.no_grad()
def save_figures(model, test_loader, latent_dim, out, device):
    model.eval()
    x = next(iter(test_loader))[0][:32].to(device)
    mu, _ = model.encode(x)
    recon = model.decode(mu).sigmoid().reshape(-1, 1, 28, 28)
    save_image(torch.cat([x, recon]), out / "reconstructions.png", nrow=8)       # top: inputs, bottom: reconstructions
    z = torch.randn(64, latent_dim, device=device)
    probs = model.decode(z).sigmoid().reshape(-1, 1, 28, 28)
    save_image(probs, out / "prior_mean_images.png", nrow=8)                     # decoder probabilities (mean images)
    save_image(torch.bernoulli(probs), out / "prior_pixel_samples.png", nrow=8)  # actual Bernoulli samples
    if latent_dim == 2:
        # decoded latent manifold: a 20 x 20 grid over [-3, 3]^2
        g = torch.linspace(-3, 3, 20)
        zz = torch.stack(torch.meshgrid(-g, g, indexing="ij"), -1).flip(-1).reshape(-1, 2).to(device)
        grid = model.decode(zz).sigmoid().reshape(-1, 1, 28, 28)
        save_image(grid, out / "latent_manifold.png", nrow=20)
        # where the test digits land (encoder means), coloured by label
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mus, labels = [], []
        for xb, yb in test_loader:
            m, _ = model.encode(xb.to(device))
            mus.append(m.cpu()); labels.append(yb)
        mus, labels = torch.cat(mus), torch.cat(labels)
        plt.figure(figsize=(6, 6))
        plt.scatter(mus[:, 0], mus[:, 1], c=labels, cmap="tab10", s=2, alpha=0.6)
        plt.colorbar(label="digit"); plt.xlabel("z1"); plt.ylabel("z2"); plt.title("Encoder means of the test set")
        plt.tight_layout(); plt.savefig(out / "latent_scatter.png", dpi=120); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.latent_dim < 1:
        parser.error("epochs, batch size and latent dimension must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(binarize)])
    train = datasets.MNIST(str(args.data), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(args.data), train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=0)
    model = VAE(args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = model(x)
            loss, _, _ = loss_terms(logits, x, mu, logvar)   # one latent sample per example
            loss.backward()
            optimizer.step()
            total += loss.item() * x.shape[0]
            count += x.shape[0]
        test_loss, test_recon, test_kl = evaluate(model, test_loader, device)
        history.append([epoch, round(total / count, 3), round(test_loss, 3), round(test_recon, 3), round(test_kl, 3)])
        print(f"epoch {epoch:02d} | train {total / count:.2f} | test loss {test_loss:.2f} "
              f"= recon {test_recon:.2f} + KL {test_kl:.2f} nats/image")
    with open(args.out / "history.csv", "w", newline="") as f:
        csv.writer(f).writerows([["epoch", "train_loss", "test_loss", "test_recon", "test_kl"]] + history)
    save_figures(model, test_loader, args.latent_dim, args.out, device)
    torch.save({"state_dict": model.state_dict(), "latent_dim": args.latent_dim,
                "binarization": "x >= 0.5", "seed": args.seed}, args.out / "vae.pt")


if __name__ == "__main__":
    main()
