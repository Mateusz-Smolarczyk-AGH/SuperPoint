import cv2
import numpy as np
import pynq_dpu

image_size = (270, 270)
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Stabilizacja numeryczna
    return exp_z / exp_z.sum(axis=0)

def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Nie udało się wczytać obrazu: {img_file}")
        return None, None
    img = cv2.resize(img, img_size)
    img_preprocessed = img.mean(-1) / 255
    print(f"Obraz {img_file} został przetworzony i wyświetlony.")
    
    return img_preprocessed

class NetworkDPU:
    
    def __init__(self, xmodel_path: str = 'MiniResNet_qu.xmodel', dpu_path: str = 'dpu.bit'):
        self.ov: pynq_dpu.DpuOverlay = pynq_dpu.DpuOverlay(dpu_path, download=True)
        self.ov.load_model(xmodel_path)
        self.dpu = self.ov.runner
        print(self.ov.runner)
        inputTensors = self.dpu.get_input_tensors()
        outputTensors = self.dpu.get_output_tensors()
        # get list of shapes
        shapeIn = np.array([it.dims for it in inputTensors])
        shapeOut = np.array([ot.dims for ot in outputTensors])
        self.shapeIn = shapeIn
        self.shapeOut = shapeOut
        self.buff_in = [np.zeros(sh, np.int8, order='C') for sh in shapeIn]
        self.buff_out = [np.zeros(sh, np.int8, order='C') for sh in shapeOut]
        
        self.input_repr = [(it.get_attr('bit_width'), it.get_attr('fix_point')) for it in inputTensors]
        self.output_repr = [(ot.get_attr('bit_width'), ot.get_attr('fix_point')) for ot in outputTensors]
    
    def input_float_to_int8(self, x: np.ndarray) -> np.ndarray:
        BIT_WIDTH, PRECISION_BITS = self.input_repr[0]
        x = x * (2**PRECISION_BITS)
        x = np.floor(x)
        x = np.clip(x,-128, 127)
        return x.astype(np.int8)
    
    def output_int8_to_float(self, y: np.ndarray):
        BIT_WIDTH, PRECISION_BITS = self.output_repr[0]
        PRECISION = 1 / 2**PRECISION_BITS
        y = y * PRECISION
        return y.astype(np.float32)
    
    def process(self, x: np.ndarray):
        x_int = int(x)
        self.buff_in[0] = x_int
        job_id, status = self.dpu.execute_async(self.buff_in, self.buff_out)
        self.dpu.wait(job_id)
        y = self.buff_out[0]
        scores, descriptors_dense = float(y)
        scores = softmax(scores)
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * 8, w * 8
        )
        return scores, descriptors_dense
    
    def __call__(self, x: np.ndarray):
        return self.process(x)

img_file = "1.ppm"
img_preprocessed = preprocess_image(img_file, image_size)
net = NetworkDPU(xmodel_path='SuperPoint_short_int.xmodel', 
                 dpu_path='dpu.bit')
print("network build done")
y_pred = net(img_preprocessed)
print("done")