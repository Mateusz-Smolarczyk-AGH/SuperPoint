import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
detection_thresh = 0.005
nms_radius = 5
import superpoint_pytorch
sp_th = superpoint_pytorch.SuperPoint(detection_threshold=detection_thresh, nms_radius=nms_radius).eval()

print("start")
print('Config:', sp_th.conf)
print(sp_th)
weights_path = "superpoint_v6_from_tf.pth"

weights = torch.load(weights_path)

print(weights.keys())  # Jeśli plik zawiera więcej danych (np. cały model, stan)
# Lub
# Wyświetl wagę przykładowej warstwy
example_layer = weights['backbone.0.0.conv.weight']
print(example_layer.shape)  # Rozmiar
print(example_layer)  # Podgląd wartości

output_file = "weights_analysis.txt"
with open(output_file, "w") as f:
    for key, tensor in weights.items():
        if isinstance(tensor, torch.Tensor):
            f.write(f"Warstwa: {key}\n")
            f.write(f"Typ danych: {tensor.dtype}\n")
            f.write(f"Shape: {list(tensor.shape)}\n")
            f.write("\n")
        else:
            # Obsługa przypadków, gdy wartość nie jest tensorem (np. scalar lub metadane)
            f.write(f"Warstwa: {key} (nie jest tensorem, typ: {type(tensor)})\n\n")

print(f"Dane zapisane w pliku: {output_file}")


from torch.fx import symbolic_trace

def analyze_flow(model, input_tensor):
    traced = symbolic_trace(model)
    x = input_tensor
    for node in traced.graph.nodes:
        if node.op == 'call_module':  # Rozpoznaj operacje na warstwach
            module = dict(model.named_modules())[node.target]
            x = module(x)
            print(f"{node.target}: typ={x.dtype}, kształt={x.shape}")
        elif node.op == 'call_function':  # Rozpoznaj operacje funkcji (np. torch.add)
            print(f"{node.target}: funkcja nie przeanalizowana")

