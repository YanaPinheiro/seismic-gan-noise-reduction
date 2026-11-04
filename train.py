import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Importar os nossos módulos locais
from generate_synthetic_data import generate_ricker_wavelet, add_noise
from models.generator import SeismicGenerator1D
from models.discriminator import SeismicDiscriminator1D

def train_gan(epochs=100, batch_size=16):
    print("A iniciar o pipeline de treino da Seismic GAN...")
    
    # Instanciar modelos
    generator = SeismicGenerator1D()
    discriminator = SeismicDiscriminator1D()
    
    # Funções de perda e Otimizadores
    criterion_GAN = nn.BCELoss()  # Perda adversarial (Discriminador/Gerador)
    criterion_pixelwise = nn.L1Loss() # Perda de reconstrução do sinal sísmico
    
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Discriminator if hasattr(optim, 'Discriminator') else optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Simulação de um lote de dados de treino
    t, clean_signal = generate_ricker_wavelet()
    clean_tensor = torch.tensor(clean_signal, dtype=torch.float32).view(1, 1, -1).repeat(batch_size, 1, 1)
    
    noise = torch.randn(batch_size, 1, 128) * 0.3
    noisy_tensor = clean_tensor + noise
    
    # Loop de treino simplificado para demonstração
    for epoch in range(5): # Pequeno teste de 5 épocas
        # -----------------
        #  Treinar Gerador
        # -----------------
        optimizer_G.zero_grad()
        
        # Gerar sinal limpo a partir do ruído
        gen_signals = generator(noisy_tensor)
        
        # Rótulos válidos para enganar o discriminador
        valid = torch.ones(batch_size, 1)
        
        g_loss_GAN = criterion_GAN(discriminator(gen_signals), valid)
        g_loss_pixel = criterion_pixelwise(gen_signals, clean_tensor)
        
        # Perda total do Gerador (GAN + Reconstrução L1)
        g_loss = g_loss_GAN + 100 * g_loss_pixel
        
        g_loss.backward()
        optimizer_G.step()
        
        print(f"Época [{epoch+1}/5] | Perda Gerador (G Loss): {g_loss.item():.4f}")

    print("Treino simulado concluído com sucesso!")

if __name__ == "__main__":
    train_gan()
