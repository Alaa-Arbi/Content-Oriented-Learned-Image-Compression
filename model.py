import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__() 
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Flatten(),            # Flatten the input
            nn.Linear(256*256*3, 256),
            nn.ReLU()
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(256, 256*256*3),
            nn.Sigmoid()  # Sigmoid activation for pixel values (assuming image in [0, 1] range)
        )
        # Discriminator layers
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*256*3, 256),
            nn.Sigmoid()  # Sigmoid activation for pixel values (assuming image in [0, 1] range)
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        x_decoded = x_decoded.view(-1,3,256,256)
        return x_decoded
