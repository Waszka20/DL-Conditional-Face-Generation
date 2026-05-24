from torch import nn
import torch


class CVAE(nn.Module):

    def __init__(self, latent_dim=64, n_classes=5, image_size=64):

        super().__init__()

        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.image_size = image_size



        self.encoder = nn.Sequential(

            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # dynamic size after encoder
        self.feature_size = image_size // 16

        self.flatten_dim = 256 * self.feature_size * self.feature_size

        # latent
        self.fc_mu = nn.Linear(self.flatten_dim + n_classes, latent_dim)

        self.fc_logvar = nn.Linear(self.flatten_dim + n_classes, latent_dim)




        self.decoder_input = nn.Linear(
            latent_dim + n_classes,
            self.flatten_dim
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 3, 3, 1, 1),

            nn.Tanh()
        )

    def encode(self, x, y):

        h = self.encoder(x)

        h = h.view(h.size(0), -1)

        h = torch.cat([h, y], dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z, y):

        h = torch.cat([z, y], dim=1)

        h = self.decoder_input(h)

        h = h.view(
            -1,
            256,
            self.feature_size,
            self.feature_size
        )

        return self.decoder(h)

    def forward(self, x, y):

        mu, logvar = self.encode(x, y)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decode(z, y)

        return x_hat, mu, logvar
    




class CVAE2(nn.Module):
    def __init__(self, latent_dim=128, n_classes=40, image_size=64):
        super().__init__()
        self.n_classes = n_classes
        self.image_size = image_size

        self.fc1 = nn.Linear(3*image_size*image_size + n_classes, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        self.fc3 = nn.Linear(latent_dim + n_classes, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc5 = nn.Linear(2048, 3*image_size*image_size)

    def encode(self, x, y):
        x = x.view(x.size(0), -1)

        h = torch.relu(self.fc1(torch.cat([x, y], dim=1)))
        h = torch.relu(self.fc2(h))

        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        h = torch.relu(self.fc3(torch.cat([z, y], dim=1)))
        h = torch.relu(self.fc4(h))

        return torch.sigmoid(self.fc5(h)).view(-1,3,self.image_size,self.image_size)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y)
        return x_hat, mu, logvar
    
