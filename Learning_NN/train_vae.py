import torch
import torchvision

# NN Optimize
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        hidden_dim = 400
        
        # Encoder
        self.enc_layer1 = nn.Linear(input_dim, hidden_dim)
        self.enc_activate1 = nn.ReLU()
        self.enc_layer2_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_layer2_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_layer1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_activate1 = nn.ReLU()
        self.dec_layer2 = nn.Linear(hidden_dim, input_dim)

        # Sigmoid for MNIST
        self.sigmoid = nn.Sigmoid()

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.rand_like(std)
        return mu + std * eps
    

    def encoder(self, data):
        latent_data = self.enc_layer1(data)
        latent_data = self.enc_activate1(latent_data)
        mu = self.enc_layer2_mu(latent_data)
        logvar = self.enc_layer2_logvar(latent_data)

        return mu, logvar


    def decoder(self, z_samples):
        # Decode
        reconst_data = self.dec_layer1(z_samples)
        reconst_data = self.dec_activate1(reconst_data)
        reconst_data = self.dec_layer2(reconst_data)

        # MNIST Change
        reconst_data = self.sigmoid(reconst_data)

        return reconst_data

    def forward(self, data):
        # Encoder
        mu, logvar = self.encoder(data)

        # Sampling in Latent Space
        z_samples = self.reparameterize(mu, logvar)

        # Decode
        reconst_data = self.decoder(z_samples)

        return reconst_data, mu, logvar
    
    def get_loss(self, data, reconst_data, mu, logvar):
        MSE = F.mse_loss(data, reconst_data, reduction="sum") # reduction?
        KLD = -torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD
    
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, data_loader, optimizer, epoch):
    for i in range(epoch):
        for data, target in data_loader:
            # Original Data : batch_size, 784
            mnist_data = data.unsqueeze(1).reshape(data.shape[0], -1).to(device)
            optimizer.zero_grad()
        
            reconst_data, mu, logvar = model(mnist_data)

            loss = model.get_loss(mnist_data, reconst_data, mu, logvar)
            loss.backward()
            optimizer.step()
        print(f"epoch : {i}/{epoch}")

    

def main():

    dataset = torchvision.datasets.MNIST('./', download=True, transform=torchvision.transforms.ToTensor())
    train_size = int(len(dataset) * 0.8)
    test_size = int(len(dataset) - train_size)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 4, shuffle=True)
    
    input_dim = int(28 * 28)
    latent_dim = 20
    model = VAE(input_dim, latent_dim).to(device)

    epoch = 30
    learning_rate = 0.001
    

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    # print(model.parameters().keys())
    train(model, train_data_loader, optimizer, epoch)

    model_save_path = 'vae_mnist_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved successfully to {model_save_path}")

    # Inference
    with torch.no_grad():
        sample_size = 64
        
        # KLD Loss --> Z ~ N(0,I)
        z_sample = torch.randn(sample_size, latent_dim).to(device)
        reconst_data = model.decoder(z_sample)
        generated_img = reconst_data.view(sample_size, 1, 28, 28)

    grid_img = torchvision.utils.make_grid(
        generated_img,
        nrow=8,
        padding=2,
        normalize=True
    )

    import matplotlib.pyplot as plt
    plt.imshow(grid_img.permute(1,2,0).cpu())
    plt.axis('off')
    plt.show()


if __name__=="__main__":
    main()

        

