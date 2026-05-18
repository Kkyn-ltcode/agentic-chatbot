# Debug: print all buffer/tensor devices in memory module
print("=== Memory internal state devices ===")
for name, buf in memory.named_buffers():
    print(f"  buffer {name}: {buf.device}")
for name, param in memory.named_parameters():
    print(f"  param  {name}: {param.device}")
print(f"  src_cpu: {src_cpu.device}")
print(f"  pos_dst_cpu: {pos_dst_cpu.device}")
