#!/usr/bin/env python3
import struct
import numpy as np


class TernaryWeightConverter:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        
    def ternarize_weights(self, weights):
        alpha = np.abs(weights).mean()
        if alpha == 0:
            return weights.astype(np.int8), 1.0
        threshold = 0.7 * alpha
        
        ternary = np.where(weights > threshold, 1.0, 
                 np.where(weights < -threshold, -1.0, 0.0))
        
        scale = alpha / 0.7
        return ternary.astype(np.int8), scale
    
    def convert(self):
        print(f"Reading {self.input_path}...")
        
        with open(self.input_path, 'rb') as fin, open(self.output_path, 'wb') as fout:
            # Copy header
            magic = fin.read(4)
            version = struct.unpack('<I', fin.read(4))[0]
            num_layers = struct.unpack('<I', fin.read(4))[0]
            print(f"Version: {version}, Layers: {num_layers}")
            
            fout.write(magic)
            fout.write(struct.pack('<I', version))
            fout.write(struct.pack('<I', num_layers))
            
            for i in range(num_layers):
                pos_before = fin.tell()
                layer_type = struct.unpack('<I', fin.read(4))[0]
                name = fin.read(64)
                
                if layer_type == 0:  # CONV
                    shape = struct.unpack('<9I', fin.read(36))
                    C_out, C_in, kH, kW = shape[0], shape[1], shape[2], shape[3]
                    groups = shape[8] if shape[8] > 0 else 1
                    C_in_g = C_in // groups
                    w_size = C_out * C_in_g * kH * kW
                    
                    # Read quant (in_zp, out_zp, is_float, out_scale)
                    quant = fin.read(16)
                    
                    # Read weight data
                    weight_bytes = fin.read(w_size)
                    pad = (4 - w_size % 4) % 4
                    fin.read(pad)
                    req_scale = fin.read(C_out * 4)
                    bias = fin.read(C_out * 8)
                    
                    # Ternarize
                    weights = np.frombuffer(weight_bytes, dtype=np.int8).astype(np.float32)
                    weights = weights.reshape(C_out, C_in_g * kH * kW)
                    ternary_w, scale = self.ternarize_weights(weights)
                    
                    name_str = name.rstrip(b'\x00').decode('latin1', errors='ignore')
                    print(f"  {name_str}: {weights.shape} -> scale={scale:.4f}")
                    
                    # Write - keep original quant (with original out_scale), but also use ternary scale
                    fout.write(struct.pack('<I', layer_type))
                    fout.write(name)
                    fout.write(struct.pack('<9I', *shape))
                    fout.write(quant)  # Original quant data
                    fout.write(ternary_w.astype(np.int8).tobytes())
                    fout.write(b'\x00' * pad)
                    # Use ternary scale for req_scale
                    new_req_scale = np.full(C_out, scale, dtype=np.float32)
                    fout.write(new_req_scale.tobytes())
                    fout.write(bias)
                    
                elif layer_type == 1:  # GEMM
                    shape = struct.unpack('<9I', fin.read(36))
                    C_out, C_in = shape[0], shape[1]
                    w_size = C_out * C_in
                    
                    quant = fin.read(16)
                    weight_bytes = fin.read(w_size)
                    pad = (4 - w_size % 4) % 4
                    fin.read(pad)
                    req_scale = fin.read(C_out * 4)
                    bias = fin.read(C_out * 8)
                    
                    weights = np.frombuffer(weight_bytes, dtype=np.int8).astype(np.float32)
                    weights = weights.reshape(C_out, C_in)
                    ternary_w, scale = self.ternarize_weights(weights)
                    
                    name_str = name.rstrip(b'\x00').decode('latin1', errors='ignore')
                    print(f"  {name_str}: {weights.shape} -> scale={scale:.4f}")
                    
                    fout.write(struct.pack('<I', layer_type))
                    fout.write(name)
                    fout.write(struct.pack('<9I', *shape))
                    fout.write(quant)
                    fout.write(ternary_w.astype(np.int8).tobytes())
                    fout.write(b'\x00' * pad)
                    new_req_scale = np.full(C_out, scale, dtype=np.float32)
                    fout.write(new_req_scale.tobytes())
                    fout.write(bias)
                    
                elif layer_type == 2:  # ADD
                    data = fin.read(24)
                    fout.write(struct.pack('<I', layer_type))
                    fout.write(name)
                    fout.write(data)
                    
                elif layer_type == 3:  # MAXPOOL
                    data = fin.read(32)
                    fout.write(struct.pack('<I', layer_type))
                    fout.write(name)
                    fout.write(data)
                    
                elif layer_type == 4:  # AVGPOOL
                    data = fin.read(16)
                    fout.write(struct.pack('<I', layer_type))
                    fout.write(name)
                    fout.write(data)
                else:
                    print(f"Unknown layer type {layer_type}, stopping")
                    break
        
        print("\nDone!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert int8 weights to ternary')
    parser.add_argument('--input', type=str, default='weights/resnet101_int8_cpp.bin')
    parser.add_argument('--output', type=str, default='weights/resnet101_ternary.bin')
    
    args = parser.parse_args()
    
    converter = TernaryWeightConverter(args.input, args.output)
    converter.convert()


if __name__ == '__main__':
    main()
