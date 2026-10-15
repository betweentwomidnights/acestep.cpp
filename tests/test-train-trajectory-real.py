#!/usr/bin/env python3
"""Replay emitted real-data batches through Carey's decoder/PEFT/AdamW factories.

This isolates multiple optimizer updates from RNG and preprocessing policies.
It is not a replay of Carey's Fabric job loop or the historical Spark run.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--carey-root', type=Path, required=True)
p.add_argument('--checkpoint-root', type=Path, required=True)
p.add_argument('--trajectory', type=Path, required=True)
p.add_argument('--adapter-precision', choices=('fp32', 'model'), default='fp32')
p.add_argument('--timestep-precision', choices=('fp32', 'model'), default='model',
               help='FP32 isolates native timestep math; model measures Carey BF16 sampling/storage')
a = p.parse_args()
spec = importlib.util.spec_from_file_location('step_parity', Path(__file__).with_name('test-train-step-parity.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
sys.path.insert(0, str(a.carey_root.resolve()))
import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from acestep.training_v2.model_loader import load_decoder_for_training
from acestep.training_v2.loss_weighting import flow_min_snr_weights
from acestep.training_v2.optim import build_optimizer

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = load_decoder_for_training(a.checkpoint_root, variant='base', device='cuda', precision='bf16')
model.decoder = PeftModel.from_pretrained(model.decoder, str(a.trajectory/'initial'), is_trainable=True,
                                        autocast_adapter_dtype=False)
model.decoder.train()
model.decoder.config.use_cache = False
model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
params = helpers.adapter_tensors(model)
initial = load_file(str(a.trajectory/'initial/adapter_model.safetensors'), device='cpu')
assert set(params) == set(initial), 'adapter inventories differ'
if a.adapter_precision == 'fp32':
    for name, value in params.items():
        value.data = initial[name].to(device='cuda', dtype=torch.float32).clone()
starting = {k:v.detach().float().cpu().numpy().copy() for k,v in params.items()}
optimizer = build_optimizer(list(params.values()), lr=3e-4, weight_decay=0.01, device_type='cuda')
groups = {'A': lambda k:'.lora_A.' in k, 'B':lambda k:'.lora_B.' in k,
          'magnitude':lambda k:'magnitude' in k}
report = {'scope':__doc__, 'adapter_precision':a.adapter_precision, 'timestep_precision':a.timestep_precision, 'torch':torch.__version__,
          'device':torch.cuda.get_device_name(), 'updates':[], 'checkpoints':{}}
tag = a.adapter_precision + ('-t-fp32' if a.timestep_precision == 'fp32' else '')
path = a.trajectory/f'torch-trajectory-{tag}.json'
if path.exists():
    raise RuntimeError('refusing to overwrite a previous trajectory report')
native_rows = [json.loads(line) for line in (a.trajectory/'metrics.jsonl').read_text().splitlines()]
assert [r['update'] for r in native_rows] == list(range(1,len(native_rows)+1))
for row in native_rows:
    started=time.monotonic()
    directory=a.trajectory/str(row['update'])
    def tensor(name):
        return torch.from_numpy(helpers.load_dump(directory/(name+'.bin'))).to(device='cuda',dtype=torch.bfloat16)
    combined=tensor('input_latents'); target=tensor('target_velocity')
    enc=tensor('encoder_hidden')
    t=torch.from_numpy(helpers.load_dump(directory/'timesteps.bin')).to(device='cuda',
         dtype=torch.float32 if a.timestep_precision == 'fp32' else torch.bfloat16)
    noisy=combined[...,128:]; context=combined[...,:128]
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr']=row['lr']  # Exact applied native schedule; previously audited factory differences remain explicit.
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        prediction=model.decoder(hidden_states=noisy,timestep=t,timestep_r=t,
            attention_mask=torch.ones(noisy.shape[:2],device='cuda',dtype=torch.bfloat16),
            encoder_hidden_states=enc,
            encoder_attention_mask=torch.ones(enc.shape[:2],device='cuda',dtype=torch.bfloat16),
            context_latents=context)[0]
        mse=torch.nn.functional.mse_loss(prediction,target,reduction='none').mean(dim=(-1,-2)).float()
        loss=(mse*flow_min_snr_weights(t,gamma=5.0)).mean().float()
    loss.backward()
    if row['update'] == 1:
        save_file({k:v.grad.detach().float().cpu().contiguous() for k,v in params.items()},
                  str(a.trajectory/f'torch-initial-gradients-{tag}.safetensors'))
    grad=float(torch.nn.utils.clip_grad_norm_(list(params.values()),1.0))
    optimizer.step()
    loss_value=float(loss.detach())
    output={'update':row['update'],'example':row['example'],'native_loss':row['loss'],
        'torch_loss':loss_value,'loss_relative_error':abs(loss_value-row['loss'])/max(abs(row['loss']),1e-30),
        'native_grad_norm':row['grad_norm'],'torch_grad_norm':grad,'lr':row['lr'],
        'forward':helpers.flat_metrics(helpers.load_dump(directory/'native_velocity.bin'),prediction.detach().float().cpu().numpy()),
        'seconds':time.monotonic()-started}
    report['updates'].append(output)
    if (directory/'updated/adapter_model.safetensors').exists():
        native={k:v.float().numpy() for k,v in load_file(str(directory/'updated/adapter_model.safetensors')).items()}
        current={k:v.detach().float().cpu().numpy().copy() for k,v in params.items()}
        report['checkpoints'][str(row['update'])]={name:helpers.aggregate_metrics(
          {k:native[k]-initial[k].float().numpy() for k in initial},
          {k:current[k]-starting[k] for k in initial},pred) for name,pred in groups.items()}
    report['peak_torch_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('[Torch-Trajectory]',json.dumps(output),flush=True)
save_file({k:v.detach().float().cpu().contiguous() for k,v in params.items()},
          str(a.trajectory/f'torch-final-{tag}.safetensors'))
print('COMPLETE',len(native_rows),'updates',flush=True)
