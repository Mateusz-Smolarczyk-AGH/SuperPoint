import torch
import numpy as np
# Wczytanie pliku .pth
weights = torch.load("superpoint_v6_from_tf.pth")

# Przykład: wylistowanie wag
for name, param in weights.items():
    print(f"Layer: {name}, Shape: {param.shape}")
    conv1_weights_np = weights[name].cpu().detach().numpy()

    # Zapisywanie do pliku binarnego (opcjonalne)
    conv1_weights_np.astype(np.float32).tofile(name + ".bin")