#!/usr/bin/env python3
"""Replay one retained trajectory update from identical native adapter weights.

Separates compute/attention precision from accumulated optimizer drift. No
optimizer step or preprocessing runs here; this is a diagnostic, not a full
Carey/Fabric parity gate. Requires local Carey checkpoints and native captures.
"""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--carey-root', type=Path, required=True)
p.add_argument('--checkpoint-root', type=Path, required=True)
p.add_argument('--trajectory', type=Path, required=True)
p.add_argument('--update', type=int, default=17)
p.add_argument('--device', default='cuda')
p.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
p.add_argument('--attention', choices=('auto', 'math'), default='math')
p.add_argument('--repeat', type=int, default=2)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--frozen-gguf', type=Path, help='Diagnostic: load exact unquantized GGUF frozen values')
p.add_argument('--gguf-python', type=Path, help='Optional directory containing the gguf Python package')
p.add_argument('--native-result', type=Path, help='Override native forward/gradients; inputs and adapters stay fixed')
p.add_argument('--trace', action='store_true', help='Capture layer boundaries and their loss gradients on the host')
p.add_argument('--local-layers', type=int, nargs='+', help='Isolate layers using native boundary, conditioning and incoming gradient')
a = p.parse_args()
if a.update < 2 or a.repeat < 1 or a.output.exists():
    p.error('update must be >=2, repeats positive, and output must be fresh')
if a.local_layers and (a.precision != 'fp32' or a.attention != 'math'):
    p.error('same-input local layer controls require --precision fp32 --attention math')

spec = importlib.util.spec_from_file_location('step_parity', Path(__file__).with_name('test-train-step-parity.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
sys.path.insert(0, str(a.carey_root.resolve()))
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from peft import PeftModel
from safetensors.torch import load_file, save_file
from acestep.training_v2.model_loader import load_decoder_for_training
from acestep.training_v2.loss_weighting import flow_min_snr_weights

torch.manual_seed(42)
torch.set_num_threads(12)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(a.attention == 'math')
dtype = torch.float32 if a.precision == 'fp32' else torch.bfloat16
# Load BF16 frozen values first, then widen them for the FP32 arithmetic control.
# Keep only the decoder resident; the captured conditioning bypasses the encoder.
model = load_decoder_for_training(a.checkpoint_root, variant='base', device='cpu', precision='bf16')
decoder = model.decoder
del model
gc.collect()
weight_audit = None
if a.frozen_gguf:
    if a.gguf_python:
        sys.path.append(str(a.gguf_python))
    from gguf import GGUFReader, GGMLQuantizationType, quants
    reader = GGUFReader(a.frozen_gguf)
    tensors = {v.name.removeprefix('decoder.'): v for v in reader.tensors if v.name.startswith('decoder.')}
    digest = hashlib.sha256()
    with a.frozen_gguf.open('rb') as frozen_file:
        for chunk in iter(lambda: frozen_file.read(1024 * 1024), b''):
            digest.update(chunk)
    weight_audit = {'different_tensors': [], 'count': 0, 'gguf_sha256': digest.hexdigest()}
    with torch.no_grad():
        for name, parameter in decoder.named_parameters():
            raw = tensors.pop(name)
            if raw.tensor_type not in (GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16):
                raise RuntimeError('this diagnostic accepts only unquantized frozen values')
            values = quants.dequantize(raw.data, raw.tensor_type).reshape(parameter.shape).copy()
            original = parameter.float().numpy()
            if not np.array_equal(original, values):
                weight_audit['different_tensors'].append({'name': name, 'metrics': h.flat_metrics(original, values)})
            parameter.copy_(torch.from_numpy(values))
            weight_audit['count'] += 1
    assert not tensors, f'unmapped GGUF decoder weights: {list(tensors)}'
    del reader
before = a.trajectory / str(a.update - 1) / 'updated'
decoder = PeftModel.from_pretrained(decoder, str(before), is_trainable=True, autocast_adapter_dtype=False)
decoder.to(device=a.device, dtype=dtype)
decoder.train()
decoder.config.use_cache = False
decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
params = h.adapter_tensors(SimpleNamespace(decoder=decoder))
state = load_file(str(before / 'adapter_model.safetensors'))
assert set(params) == set(state)
for name, value in params.items():
    value.data = state[name].to(device=a.device, dtype=torch.float32).clone()
directory = a.trajectory / str(a.update)
native_result = a.native_result or directory
native_gradients = {k: v.float().numpy() for k, v in load_file(str(native_result / 'native_gradients.safetensors')).items()}
def tensor(name, kind=dtype):
    return torch.from_numpy(h.load_dump(directory / (name + '.bin'))).to(device=a.device, dtype=kind)
combined = tensor('input_latents')
encoder = tensor('encoder_hidden')
target = tensor('target_velocity')
t = tensor('timesteps', torch.float32)
native_prediction = h.load_dump(native_result / 'native_velocity.bin')
report = {'scope': __doc__, 'torch': torch.__version__, 'precision': a.precision,
          'attention': a.attention, 'device': a.device, 'update': a.update, 'runs': [],
          'hardware': torch.cuda.get_device_name() if a.device.startswith('cuda') else 'CPU',
          'frozen_weight_audit': weight_audit,
          'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in [before / 'adapter_model.safetensors',
                                        native_result / 'native_gradients.safetensors', native_result / 'native_velocity.bin',
                                        *sorted(directory.glob('*.bin'))]}}
groups = {'A': lambda k: '.lora_A.' in k, 'B': lambda k: '.lora_B.' in k,
          'magnitude': lambda k: 'magnitude' in k}
captures = {}
handles = []
if a.trace:
    combined.requires_grad_(True)  # Also expose the first frozen projection's output gradient.
    base = decoder.get_base_model()
    def host(value):
        return value.detach().float().cpu().numpy().copy()
    def boundary(index, value):
        key = f'boundary-{index}'
        if key not in captures:
            captures[key] = host(value)
            if value.requires_grad:
                value.register_hook(lambda gradient: captures.__setitem__(f'gradient-{index}', host(gradient)))
    for index, layer in enumerate(base.layers):
        def before_layer(module, args, kwargs, index=index):
            boundary(index, kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0])
        handles.append(layer.register_forward_pre_hook(before_layer, with_kwargs=True))
    handles.append(base.layers[-1].register_forward_hook(lambda module, args, output:
                   boundary(len(base.layers), output[0])))
    def time_hook(label):
        def hook(module, args, output):
            captures[f'temb-{label}'] = host(output[0])
            captures[f'tproj-{label}'] = host(output[1])
        return hook
    handles.append(base.time_embed.register_forward_hook(time_hook('t')))
    handles.append(base.time_embed_r.register_forward_hook(time_hook('r')))
    handles.append(base.condition_embedder.register_forward_hook(lambda module, args, output:
                   captures.__setitem__('condition', host(output))))
previous = None
for repeat in range(a.repeat):
    captures.clear()
    decoder.zero_grad(set_to_none=True)
    with (sdpa_kernel(SDPBackend.MATH) if a.attention == 'math' else nullcontext()):
        with (torch.autocast('cuda', dtype=dtype) if a.device.startswith('cuda') and dtype != torch.float32 else nullcontext()):
            prediction = decoder(hidden_states=combined[..., 128:], context_latents=combined[..., :128],
                                 timestep=t, timestep_r=t,
                                 attention_mask=torch.ones(combined.shape[:2], device=a.device, dtype=dtype),
                                 encoder_hidden_states=encoder,
                                 encoder_attention_mask=torch.ones(encoder.shape[:2], device=a.device, dtype=dtype))[0]
            mse = torch.nn.functional.mse_loss(prediction, target, reduction='none').mean(dim=(-1, -2)).float()
            loss = (mse * flow_min_snr_weights(t, gamma=5.0)).mean().float()
        loss.backward()
    actual = prediction.detach().float().cpu().numpy().copy()
    gradients = {k: v.grad.detach().float().cpu().numpy().copy() for k, v in params.items()}
    report['runs'].append({'loss': float(loss.detach()), 'forward': h.flat_metrics(native_prediction, actual),
                           'gradients': {name: h.aggregate_metrics(native_gradients, gradients, select)
                                         for name, select in groups.items()},
                           'repeat_bitexact': None if previous is None else
                               np.array_equal(previous[0], actual) and
                               all(np.array_equal(previous[1][k], gradients[k]) for k in gradients)})
    previous = (actual, gradients)
    if a.device.startswith('cuda'):
        report['peak_torch_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if repeat == 0:
        save_file({k: torch.from_numpy(v.copy()) for k, v in gradients.items()},
                  str(a.output.with_suffix('.safetensors')))
        np.save(a.output.with_suffix('.npy'), actual)
        if a.trace:
            captures['temb'] = captures['temb-t'] + captures['temb-r']
            captures['tproj'] = captures['tproj-t'] + captures['tproj-r']
            np.savez(a.output.with_suffix('.trace.npz'), **captures)
            report['layer_trace'] = {name: h.flat_metrics(h.load_dump(native_result/'trace'/(name+'.bin')), value)
                                    for name, value in captures.items()
                                    if (native_result/'trace'/(name+'.bin')).exists()}
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report['runs'][-1]), flush=True)

if a.local_layers:
    for handle in handles:
        handle.remove()
    base = decoder.get_base_model()
    base.gradient_checkpointing_disable()
    def native_tensor(name):
        return torch.from_numpy(h.load_dump(native_result/'trace'/(name+'.bin'))).to(device=a.device, dtype=dtype)
    native_condition = native_tensor('condition')
    native_tproj = native_tensor('tproj').reshape(1, 6, -1)
    report['local_layers'] = {}
    for index in a.local_layers:
        if not 0 <= index < len(base.layers):
            raise ValueError(f'layer index outside decoder: {index}')
        decoder.zero_grad(set_to_none=True)
        x = native_tensor(f'boundary-{index}').requires_grad_(True)
        incoming = native_tensor(f'gradient-{index+1}')
        positions = torch.arange(x.shape[1], device=a.device).unsqueeze(0)
        rotary = base.rotary_emb(x, positions)
        mask = None
        window = base.config.sliding_window
        if base.layers[index].attention_type == 'sliding_attention' and x.shape[1] > window:
            distance = (positions[:, :, None] - positions[:, None, :]).abs()
            mask = torch.zeros((1, 1, x.shape[1], x.shape[1]), device=a.device, dtype=dtype)
            mask.masked_fill_(distance[:, None] > window, float('-inf'))
        with sdpa_kernel(SDPBackend.MATH):
            output = base.layers[index](hidden_states=x, position_embeddings=rotary, temb=native_tproj,
                                        encoder_hidden_states=native_condition, attention_mask=mask,
                                        encoder_attention_mask=None)[0]
            (output * incoming).sum().backward()
        layer_key = f'.layers.{index}.'
        local_gradients = {k: v.grad.detach().float().cpu().numpy().copy()
                           for k, v in params.items() if layer_key in k}
        selected_native = {k: v for k, v in native_gradients.items() if layer_key in k}
        assert selected_native and set(selected_native) == set(local_gradients)
        report['local_layers'][str(index)] = {
            'forward': h.flat_metrics(h.load_dump(native_result/'trace'/f'boundary-{index+1}.bin'),
                                     output.detach().float().cpu().numpy()),
            'input_gradient': h.flat_metrics(h.load_dump(native_result/'trace'/f'gradient-{index}.bin'),
                                            x.grad.detach().float().cpu().numpy()),
            'adapter_gradients': {name: h.aggregate_metrics(selected_native, local_gradients, select)
                                  for name, select in groups.items()}}
        a.output.write_text(json.dumps(report, indent=2, allow_nan=False))
        print('LOCAL', index, json.dumps(report['local_layers'][str(index)]), flush=True)
    # Close the remaining output-head boundary with identical hidden states and
    # timestep modulation, using the same real loss rather than a synthetic VJP.
    x = native_tensor(f'boundary-{len(base.layers)}').requires_grad_(True)
    modulation = native_tensor('temb').reshape(1, 1, -1) + base.scale_shift_table
    shift, scale = modulation.chunk(2, dim=1)
    tail_prediction = base.proj_out(base.norm_out(x) * (1 + scale) + shift)
    tail_loss = (torch.nn.functional.mse_loss(tail_prediction, target, reduction='none').mean(dim=(-1, -2)) *
                 flow_min_snr_weights(t, gamma=5.0)).mean()
    tail_loss.backward()
    report['local_tail'] = {'forward': h.flat_metrics(native_prediction, tail_prediction.detach().float().cpu().numpy()),
                            'input_gradient': h.flat_metrics(
                                h.load_dump(native_result/'trace'/f'gradient-{len(base.layers)}.bin'),
                                x.grad.detach().float().cpu().numpy()), 'loss': float(tail_loss.detach())}
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print('TAIL', json.dumps(report['local_tail']), flush=True)
