memory = memory.to(cpu)
memory.reset_state()

# Force ALL internal plain tensors to CPU (TGN stores these outside nn.Module buffer system)
for attr in ['memory', 'last_update', '_assoc']:
    if hasattr(memory, attr):
        val = getattr(memory, attr)
        if isinstance(val, torch.Tensor):
            setattr(memory, attr, val.to(cpu))
        elif hasattr(val, 'data'):  # nn.Parameter-like
            val.data = val.data.to(cpu)

# Update memory & neighbor loader (CPU)
memory.update_state(src_cpu, pos_dst_cpu, t_cpu, msg_cpu)
neighbor_loader.insert(src_cpu, pos_dst_cpu)

# Re-anchor memory internals to CPU after update (guards against TGN internal device drift)
for attr in ['memory', 'last_update', '_assoc']:
    if hasattr(memory, attr):
        val = getattr(memory, attr)
        if isinstance(val, torch.Tensor):
            setattr(memory, attr, val.to(cpu))
