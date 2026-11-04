import numpy as np
import matplotlib.pyplot as plt

def generate_ricker_wavelet(freq=25.0, length=0.128, dt=0.001):
    """Gera uma onda de Ricker para simular o sinal sísmico limpo."""
    t = np.arange(-length/2, length/2, dt)
    y = (1.0 - 2.0 * (np.pi ** 2) * (freq ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (freq ** 2) * (t ** 2))
    return t, y

def add_noise(signal, noise_level=0.3):
    """Adiciona ruído gaussiano ao sinal sísmico."""
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

if __name__ == "__main__":
    print("A gerar dados sísmicos sintéticos...")
    t, clean_signal = generate_ricker_wavelet()
    noisy_signal = add_noise(clean_signal)
    
    print("Sinal gerado com sucesso!")
    print(f"Tamanho do vetor de tempo: {len(t)} pontos")
