"""Opt-in focused native/Carey comparison using the actual downstream Python sources.

No model loads: replay controlled inputs through FixedLoRAModule.training_step
with a recorded prediction. This isolates batch math/loss derivatives, then
checks real AdamW factories and schedules. Known policy differences are reported
explicitly and asserted, rather than accepted as numerical parity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import warnings

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--native-exe', type=Path, required=True)
p.add_argument('--carey-root', type=Path, required=True, help='services/carey containing acestep')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
fixture_path = a.output.resolve() / 'native.json'
subprocess.run([str(a.native_exe.resolve()), str(fixture_path)], check=True)
f = json.loads(fixture_path.read_text())
sys.path.insert(0, str(a.carey_root.resolve()))
import torch
from acestep.training_v2 import fixed_lora_module as fixed
from acestep.training_v2.optim import build_optimizer, build_scheduler
from acestep.training_v2.timestep_sampling import sample_timesteps
from acestep.training.data_module import collate_preprocessed_batch
from acestep.training_v2 import trainer_fixed as trainer_module

def tensor(values, *shape):
    return torch.tensor(values, dtype=torch.float32).reshape(shape) if shape else torch.tensor(values, dtype=torch.float32)

def close(x, y, atol=2e-6, rtol=2e-5):
    torch.testing.assert_close(torch.as_tensor(x), torch.as_tensor(y), atol=atol, rtol=rtol, check_dtype=False)

report = {'scope': 'controlled CPU FP32 batch/loss/optimizer helpers; no full DiT or preprocessing models',
          'torch': torch.__version__, 'batches': [], 'known_differences': {}}
files = ['training_v2/fixed_lora_module.py', 'training_v2/loss_weighting.py',
         'training_v2/timestep_sampling.py', 'training_v2/optim.py',
         'training_v2/trainer_fixed.py', 'training/data_module.py']
report['carey_sources'] = {name: hashlib.sha256((a.carey_root / 'acestep' / name).read_bytes()).hexdigest() for name in files}

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prediction = torch.nn.Parameter(torch.full((2, 4, 2), 0.125))
        self.inputs = None
    def forward(self, **kwargs):
        self.inputs = kwargs
        return (self.prediction,)

for c in f['batches']:
    decoder = Decoder()
    module = fixed.FixedLoRAModule.__new__(fixed.FixedLoRAModule)
    torch.nn.Module.__init__(module)
    module.model = torch.nn.Module()
    module.model.decoder = decoder
    module.device = torch.device('cpu')
    module.device_type = 'cpu'
    module.dtype = torch.float32
    module.transfer_non_blocking = False
    module._null_cond_emb = tensor(c['null'], 1, 1, 4)
    module._cfg_ratio = c['cfg']
    module._loss_weighting = 'min_snr' if c['min_snr'] else 'none'
    module._snr_gamma = 5
    module._timestep_mu, module._timestep_sigma, module._data_proportion = -0.4, 1, 0
    module.force_input_grads_for_checkpointing = False
    module.training_losses = []
    data = tensor(c['data'], 2, 4, 2)
    noise = tensor(c['target'], 2, 4, 2) + data
    t, r = tensor(c['t']), tensor(c['r'])
    masks = torch.ones((2, 4))
    if c['padded']: masks[1, 2:] = 0
    batch = dict(target_latents=data, attention_mask=masks,
                 encoder_hidden_states=tensor(c['encoder'], 2, 3, 4),
                 encoder_attention_mask=torch.ones((2, 3)), context_latents=tensor(c['context'], 2, 4, 4))
    with patch.object(fixed.torch, 'randn_like', return_value=noise), patch.object(fixed, 'sample_timesteps', return_value=(t, r)):
        loss = module.training_step(batch)
    loss.backward()
    native_input = tensor(c['input'], 2, 4, 6)
    close(decoder.inputs['hidden_states'], native_input[..., 4:])
    close(decoder.inputs['context_latents'], native_input[..., :4])
    close(decoder.inputs['encoder_hidden_states'], tensor(c['condition'], 2, 3, 4))
    close(decoder.inputs['timestep_r'], t)
    item = dict(min_snr=c['min_snr'], cfg=c['cfg'], padded=c['padded'], native_loss=c['loss'], carey_loss=loss.item())
    if not c['padded']:
        close(loss.detach(), c['loss'])
        close(decoder.prediction.grad, tensor(c['gradient'], 2, 4, 2))
        item['loss_and_gradient_match'] = True
    else:
        native_grad = tensor(c['gradient'], 2, 4, 2)
        assert native_grad[1, 2:].count_nonzero() == 0
        assert decoder.prediction.grad[1, 2:].count_nonzero() > 0
        assert abs(loss.item() - c['loss']) > 1e-4
        report['known_differences']['padding_loss'] = item
    report['batches'].append(item)

# Test Carey's actual two draws/max/r=t helper with shared normal values.
logits = tensor([-10, -2, -0.4, 0, 2, 10])
with patch('torch.randn', side_effect=[logits, logits.flip(0)]):
    t, r = sample_timesteps(6, 'cpu', torch.float32, timestep_mu=0, timestep_sigma=1)
close(t, f['timesteps']); close(r, t)
report['two_draw_timestep_match'] = True

# Production native accumulation, global clipping and three AdamW updates,
# including nonzero A/B/DoRA gradients and optimizer moments.
for c in f['optimizer']:
    params = [torch.nn.Parameter(tensor([0.125, -0.25, 0.5])), torch.nn.Parameter(tensor([0, 0.2]))]
    if c['dora']: params.append(torch.nn.Parameter(tensor([1, 0.75])))
    opt = build_optimizer(params, lr=0.001, weight_decay=0.01, device_type='cpu')
    for s in c['steps']:
        for param, key in zip(params, ['sum_a', 'sum_b', 'sum_m']):
            param.grad = tensor(s[key]) / 3
        norm = torch.nn.utils.clip_grad_norm_(params, 0.5)
        close(norm, s['norm'])
        opt.step()
        for param, key in zip(params, ['a', 'b', 'm']): close(param.detach(), s[key])
        close(opt.state[params[0]]['exp_avg'], s['first_a'])
        close(opt.state[params[0]]['exp_avg_sq'], s['second_a'])
report['three_update_lora_dora_adamw_match'] = True

def schedule(name, total, warmup):
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = build_optimizer([param], lr=0.001, device_type='cpu')
    sched = build_scheduler(opt, name, total_steps=total, warmup_steps=warmup, lr=0.001, n_restarts=4)
    lrs = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for _ in range(total):
            lrs.append(opt.param_groups[0]['lr'])
            param.grad = torch.zeros_like(param); opt.step(); sched.step()
    return lrs

for i, name in enumerate(['constant', 'linear', 'cosine', 'cosine_restarts']):
    lrs = schedule(name, 23, 2)
    if name != 'cosine_restarts': close(lrs, f['schedules'][i], atol=1e-9)
    else:
        assert max(abs(x-y) for x, y in zip(lrs, f['schedules'][i])) > 1e-4
        report['known_differences']['restart_cycle_rounding'] = {'native': f['schedules'][i], 'carey': lrs}
warm = schedule('cosine', 20, 50)
assert max(abs(x-y) for x, y in zip(warm, f['long_warmup'])) > 1e-4
report['known_differences']['warmup_clamp'] = {'native': f['long_warmup'], 'carey': warm}
zero = schedule('cosine', 20, 0)
assert abs(zero[0] - 0.0001) < 1e-9 and abs(f['zero_warmup'][0] - 0.001) < 1e-9
report['known_differences']['zero_warmup'] = {'native': f['zero_warmup'], 'carey': zero}
report['matched_schedules_with_same_effective_warmup'] = ['constant', 'linear', 'cosine']

samples = []
for c in f['collation']:
    data, context, encoder = tensor(c['data'], -1, 2), tensor(c['context'], -1, 4), tensor(c['encoder'], -1, 4)
    samples.append(dict(metadata={}, target_latents=data, context_latents=context, encoder_hidden_states=encoder,
                        attention_mask=torch.ones(data.shape[0]), encoder_attention_mask=torch.ones(encoder.shape[0])))
collated = collate_preprocessed_batch(samples)
for i, c in enumerate(f['collation']):
    close(collated['target_latents'][i], tensor(c['collated_data'], 4, 2))
    close(collated['context_latents'][i, :samples[i]['context_latents'].shape[0]], samples[i]['context_latents'])
    close(collated['encoder_hidden_states'][i, :samples[i]['encoder_hidden_states'].shape[0]], samples[i]['encoder_hidden_states'])
assert collated['context_latents'][1, 2:].count_nonzero() == 0
assert tensor(f['collation'][1]['collated_context'], 4, 4)[2:].count_nonzero() > 0
assert collated['encoder_hidden_states'][1, 2:].count_nonzero() == 0
assert tensor(f['collation'][1]['collated_encoder'], 3, 4)[2:].count_nonzero() > 0
report['known_differences']['padding_values'] = 'Carey uses zeros for context/encoder; native uses silence+mask/null condition. Valid regions and target zero padding match.'

# Execute the actual Fabric loop with a scalar differentiable model and a small
# synchronous Fabric stand-in. This measures accumulation before clipping;
# checkpoint/TensorBoard side effects and distributed execution are excluded.
norms = []
class Fabric:
    def __init__(self, **kwargs): pass
    def launch(self): pass
    def setup(self, model, opt): return model, opt
    def backward(self, loss): loss.backward()
    def clip_gradients(self, model, opt, max_norm):
        norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)))

class Logger:
    def __init__(self, *args): pass
    def __getattr__(self, name): return lambda *args, **kwargs: None

class ScalarModule:
    def __init__(self):
        self.model = torch.nn.Module()
        self.model.decoder = torch.nn.Linear(1, 1, bias=False)
        self.model.decoder.weight.data.zero_()
        self.device_type, self.device, self.dtype = 'cpu', torch.device('cpu'), torch.float32
        self.training_losses = []
    def training_step(self, batch):
        loss = self.model.decoder.weight.sum() * batch + 2
        self.training_losses.append(loss.item())
        return loss

cfg = SimpleNamespace(output_dir=str(a.output / 'loop'), effective_log_dir=str(a.output / 'tb'),
    learning_rate=0.001, weight_decay=0.01, gradient_accumulation_steps=2, max_epochs=1,
    warmup_steps=0, max_grad_norm=100, resume_from=None, log_every=1, log_heavy_every=100,
    save_every_n_epochs=100, gradient_checkpointing=False, offload_encoder=False,
    vram_preflight=False, save_best=False)
owner = SimpleNamespace(training_config=cfg, module=ScalarModule(), adapter_type='lora', _save_final=lambda path: None)
with patch.object(trainer_module, 'Fabric', Fabric, create=True), patch.object(trainer_module, 'TrainingLogger', Logger), patch('torch.cuda.is_available', return_value=False):
    updates = list(trainer_module.FixedLoRATrainer._train_fabric(owner, SimpleNamespace(train_dataloader=lambda: [1.0, 1.0, 1.0]), None))
close(norms, [1.0, 0.5])
assert updates[-1].kind == 'complete'
report['known_differences']['partial_accumulation'] = {'carey_preclip_norms': norms,
    'native_policy': 'Averages by actual example count; the same scalar gradients give [1, 1]. Carey divides the final singleton by configured accumulation=2.'}
report['passed_focused_checks'] = True
(a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS: focused math checks; known policy differences remain:', ', '.join(report['known_differences']))
