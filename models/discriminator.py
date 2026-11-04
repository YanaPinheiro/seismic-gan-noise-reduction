import torch
import torch.nn as nn

class SeismicDiscriminator1D(nn.Module):
    """
    1D Convolutional Discriminator for Seismic GAN.
    Classifies a seismic trace as real (clean) or fake (denoised).
    """
    def __init__(self):
        super(SeismicDiscriminator1D, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1), # Reduz para extrair características globais
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Saída entre 0 (falso) e 1 (real)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Teste rápido de integridade do Discriminador
    model = SeismicDiscriminator1D()
    dummy_input = torch.randn(1, 1, 128) # Batch size: 1, Canal: 1, Pontos: 128
    output = model(dummy_input)
    
    print("Modelo Discriminador PyTorch instanciado com sucesso!")
    print(f"Dimensão de saída (Probabilidade): {output.shape}")
