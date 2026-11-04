 import numpy as np
import matplotlib.pyplot as plt
import os

def generate_ricker_wavelet(freq=25.0, length=0.128, dt=0.001):
    """Gera uma onda de Ricker para simular o sinal sísmico limpo."""
    t = np.arange(-length/2, length/2, dt)
    y = (1.0 - 2.0 * (np.pi ** 2) * (freq ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (freq ** 2) * (t ** 2))
    return t, y

def add_noise(signal, noise_level=0.3):
    """Adiciona ruído gaussiano ao sinal sísmico."""
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

def plot_and_save(t, clean, noisy, output_path="data/seismic_trace_example.png"):
    """Gera e guarda o gráfico comparativo entre o sinal limpo e com ruído."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(8, 4))
    plt.plot(t, noisy, label="Noisy Trace", color="red", alpha=0.6)
    plt.plot(t, clean, label="Clean Ricker Wavelet", color="black", linewidth=2)
    plt.title("Synthetic Seismic Trace with Gaussian Noise")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Gráfico guardado com sucesso em: {output_path}")

if __name__ == "__main__":
    print("A gerar dados sísmicos sintéticos...")
    t, clean_signal = generate_ricker_wavelet()
    noisy_signal = add_noise(clean_signal)
    
    plot_and_save(t, clean_signal, noisy_signal)
    print("Processo concluído!")
