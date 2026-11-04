import torch
import torch.nn as nn

class SeismicGenerator1D(nn.Module):
    """
    1D Autoencoder Generator for Seismic Signal Noise Attenuation.
    Takes noisy 1D seismic trace as input and outputs a cleaned trace.
    """
    def __init__(self):
        super(SeismicGenerator1D, self).__init__()
        
        # Encoder: Reduz a dimensão e extrai características do sinal
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        
        # Decoder: Reconstrói o sinal limpo a partir dos recursos aprendidos
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == "__main__":
    # Teste rápido de integridade da rede
    model = SeismicGenerator1D()
    dummy_input = torch.randn(1, 1, 128) # Batch size: 1, Canal: 1, Pontos no tempo: 128
    output = model(dummy_input)
    
    print("Modelo Gerador PyTorch instanciado com sucesso!")
    print(f"Dimensão de entrada: {dummy_input.shape}")
    print(f"Dimensão de saída: {output.shape}")
