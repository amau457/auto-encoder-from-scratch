import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class encoder(nn.Module):
    def __init__(self, H, W, dim):
        super().__init__()
        #H, W sizes of input image
        #dim:  dimension of the encoded vector
        self.H = H
        self.W = W
        self.N = self.H*self.W
        self.dim = dim
        # we take the flatten image as input
        self.mlp = nn.Sequential(
            nn.Linear(self.N, 2*self.N),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.N, self.dim),
            nn.Tanh()
            )

    def forward(self, x):
        return(self.mlp.forward(x))

class decoder(nn.Module):
    def __init__(self, H, W, dim):
        super().__init__()
        #H, W sizes of output image
        #dim:  dimension of the encoded vector
        self.H = H
        self.W = W
        self.N = self.H*self.W
        self.dim = dim
        # we take the flatten image as input
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, 2*self.N),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.N, self.N), 
            nn.Sigmoid())

    def forward(self, x):
        return(self.mlp.forward(x))

class encoder_decoder(nn.Module):
    def __init__(self, H, W, dim):
        super().__init__()
        self.encoder = encoder(H, W, dim)
        self.decoder = decoder(H, W, dim)
    
    def forward(self, x):
        return(self.decoder.forward(self.encoder.forward(x)))
    
def train(model, epochs, loader, device, lr):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for x, _ in loader:
            x = x.to(device)  # x is already flattened by transform
            optimizer.zero_grad()
            y_out = model(x)
            loss = criterion(y_out, x)  # target is the input itself
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            avg = total_loss / total_samples
            print(f"Epoch {epoch+1}/{epochs} - avg reconstruction loss: {avg:.6f}")

def unnormalize_flat(tensor_flat, H, W):
    #tensor H*W to array HxW in [0,1]
    t = tensor_flat.detach().cpu()
    if t.ndim == 1:
        t = t.view(1, H, W)
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    t = t.clamp(0, 1)
    return(t.numpy())

def show_reconstructions(model, test_loader, device, n=8):
    model.to(device)
    model.eval()
    H = model.encoder.H
    W = model.encoder.W
    originals = []
    recons = []
    collected = 0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            y = model(x)
            print(model.encoder.forward(x))
            B = x.size(0)
            for i in range(B):
                originals.append(unnormalize_flat(x[i], H, W))
                recons.append(unnormalize_flat(y[i], H, W))
                collected += 1
                if collected >= n:
                    break
            if collected >= n:
                break

    plt.figure(figsize=(n * 2.2, 4.6))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(originals[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"orig #{i}")
        ax.axis('off')

        ax = plt.subplot(2, n, n + i + 1)
        plt.imshow(recons[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"recon #{i}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    epochs = 100
    lr = 1e-3
    transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Lambda(lambda x: x.view(-1)) #flatten
                                       ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    H, W = train_dataset.data[0].shape
    dim = 16 # to finetune 
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = encoder_decoder(H, W, dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, epochs, train_loader, device=device, lr=lr)
    torch.save(model, "autoencoder.pt")
    show_reconstructions(model, test_loader, device, n=8)


if __name__ == "__main__":
    main()