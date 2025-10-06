import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class encoder(nn.Module):
    def __init__(self, H, W, dim, nc = 1, ndf= 64):
        super().__init__()
        #H, W sizes of input image
        #dim:  dimension of the encoded vector
        #nc = nb of channels
        #ndf = number of filters
        self.H = H
        self.W = W
        self.N = self.H*self.W
        self.dim = dim
        self.nc = nc
        self.ndf = ndf
        # we take the image as input
        self.cnn = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14 -> 7x7
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #7x7 -> dim
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(ndf * 2, dim, kernel_size=1, bias=False),
            #nn.Tanh() 
        )

    def forward(self, x):
        return(self.cnn.forward(x))

class decoder(nn.Module):
    def __init__(self, H, W, dim, nc =1, nf = 64):
        super().__init__()
        #H, W sizes of output image
        #dim:  dimension of the encoded vector
        self.H = H
        self.W = W
        self.N = self.H*self.W
        self.dim = dim
        # we take the flatten image as input
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(dim, nf * 4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return(self.cnn.forward(x))

class encoder_decoder(nn.Module):
    def __init__(self, H, W, dim, nc = 1, nf= 64):
        super().__init__()
        self.encoder = encoder(H, W, dim, nc, nf)
        self.decoder = decoder(H, W, dim, nc, nf)
    
    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return(self.decoder(z))
    
def train(model, epochs, loader, device, lr):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for x, _ in loader:
            x = x.to(device) 
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
    t = (t + 1.0) / 2.0
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

def show_latent(model, test_loader, device, n=1000, block=True):
    model.to(device)
    model.eval()
    H = model.encoder.H
    W = model.encoder.W
    collected = 0
    res_x, res_y = [[] for _ in range(10)], [[] for _ in range(10)]

    with torch.no_grad():
        for x, label in test_loader:
            x = x.to(device)
            y = model.encoder.forward(x)
            B = x.size(0)
            for i in range(B):
                res_x[label[i]].append(y[i][0].detach().cpu().item())
                res_y[label[i]].append(y[i][1].detach().cpu().item())
                collected += 1
                if collected >= n:
                    break
            if collected >= n:
                break

    plt.figure()
    colors = ['red', 'blue', 'green', 'yellow', 'pink', 'gray', 'cyan', 'black', 'orange', 'magenta']
    for i in range(10):
        plt.plot(res_x[i], res_y[i], 'x', color=colors[i], label = str(i))
    plt.legend()
    plt.show(block=block)

def show_pca(model, test_loader, device, n=10000, block=True):
    # apply PCA to latent vectors to reduce the dimension to 2 in order to plot it
    model.to(device)
    model.eval()

    collected = 0
    ys = []
    labels = []

    with torch.no_grad():
        for x, label in test_loader:
            x = x.to(device)
            y = model.encoder.forward(x)
            B = y.size(0)
            y_flat = y.view(B, -1)   
            ys.append(y_flat.detach().cpu().numpy())
            labels.append(label.numpy())
            collected += B
            if collected >= n:
                break

    ys = np.vstack(ys)[:n]
    labels = np.concatenate(labels)[:n]

    # center the ys
    ys_centered = ys - ys.mean(axis=0, keepdims=True)

    #SVD PCA
    U, S, Vt = np.linalg.svd(ys_centered, full_matrices=False)
    components = Vt[:2]
    ys_pca = ys_centered.dot(components.T)

    num_classes = int(labels.max()) + 1
    res_x = [[] for _ in range(num_classes)]
    res_y = [[] for _ in range(num_classes)]

    for i in range(ys_pca.shape[0]):
        lbl = int(labels[i])
        res_x[lbl].append(ys_pca[i, 0])
        res_y[lbl].append(ys_pca[i, 1])

    # Plot
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'yellow', 'pink', 'gray', 'cyan', 'black', 'orange', 'magenta']
    for i in range(num_classes):
        c = colors[i % len(colors)]
        plt.plot(res_x[i], res_y[i], 'x', color=c, label=str(i))
    plt.title("PCA of latent vectors")
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show(block=block)

def main():
    epochs = 20
    lr = 1e-4
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    H, W = train_dataset.data[0].shape
    dim = 10 # to finetune 
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = encoder_decoder(H, W, dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train(model, epochs, train_loader, device=device, lr=lr)
    torch.save(model, "autoencoder.pt")
    show_latent(model, test_loader, device, n=10000, block=False)
    show_pca(model, test_loader, device, n=10000, block=True)
    show_reconstructions(model, test_loader, device, n=8)


if __name__ == "__main__":
    main()