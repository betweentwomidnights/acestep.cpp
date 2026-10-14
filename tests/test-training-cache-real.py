"""Opt-in real-model cache parity/invalidation regression. Outputs need a fresh directory."""
import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import struct
import subprocess
import time
import wave

p = argparse.ArgumentParser(description=__doc__)
for flag in ('bin-dir', 'models', 'backend', 'output'):
    p.add_argument('--' + flag, required=True)
p.add_argument('--baseline-bin-dir', help='Optional older executable for unchanged-algorithm parity')
p.add_argument('--adapter-type', choices=('lora', 'dora-rows'), default='lora')
p.add_argument('--expected-device', default='NVIDIA')
p.add_argument('--quick', action='store_true', help='Only baseline/cold/warm and mid-epoch resume')
a = p.parse_args()
root = Path(a.output).resolve()
root.mkdir(parents=True, exist_ok=False)
dataset = root / 'dataset'
dataset.mkdir()
captions = ['Quiet piano.', 'Sustained instrumental tones with soft percussion.',
    'Gentle sustained electronic tones, warm chords, soft percussion and evolving textures. ' * 4]
for index, caption in enumerate(captions):
    with wave.open(str(dataset / f'{index}.wav'), 'wb') as audio:
        audio.setparams((2, 2, 48000, 0, 'NONE', 'not compressed'))
        audio.writeframes(b''.join(struct.pack('<hh', round(1000 * math.sin(2 * math.pi * (220 + index*20) * i/48000)),
            round(900 * math.sin(2 * math.pi * 330 * i/48000))) for i in range(48000*4)))
    (dataset / f'{index}.txt').write_text(f'caption: {caption}\ngenre: ambient\nbpm: 120\n'
        'key: C major\nsignature: 4\nis_instrumental: true\ncustom_tag: smoke\nlyrics: [Instrumental]\n')
env = {k: v for k, v in os.environ.items() if not k.startswith('ACE_')}
env['GGML_BACKEND'] = a.backend
suffix = '.exe' if os.name == 'nt' else ''
binary = Path(a.bin_dir).resolve()
props = json.loads(subprocess.run([binary / ('ace-server' + suffix), '--props', '--device', a.backend],
    env=env, capture_output=True, text=True, check=True).stdout)
device = next(d for d in props['devices'] if d['name'] == a.backend)
assert props['hardware']['selected_device'] == a.backend and a.expected_device.lower() in device['description'].lower()
(root / 'props.json').write_text(json.dumps(props, indent=2))
common = ['--models', str(Path(a.models).resolve()), '--dataset', str(dataset),
    '--dit', 'acestep-v15-xl-base-Q4_K_M.gguf', '--text-encoder', 'Qwen3-Embedding-0.6B-Q8_0.gguf', '--vae', 'vae-BF16.gguf',
    '--adapter-type', a.adapter_type, '--profile', 'balanced', '--rank', '8', '--alpha', '16', '--epochs', '2',
    '--batch-size', '1', '--accumulation', '2', '--warmup-steps', '0', '--checkpoint-every', '0',
    '--window-min-seconds', '2', '--window-max-seconds', '2', '--seed', '42']
cache = root / 'cache'
results = {}

def run(label, extra=(), baseline=False):
    output = root / label
    exe_dir = Path(a.baseline_bin_dir).resolve() if baseline and a.baseline_bin_dir else binary
    options = [] if baseline and a.baseline_bin_dir else (['--preprocess-cache', 'false'] if baseline else
        ['--preprocess-cache-dir', str(cache), '--cache-memory-mb', '0'])
    command = [str(exe_dir / ('ace-train' + suffix))] + common + options + ['--output', str(output)] + list(map(str, extra))
    print(label, flush=True)
    start = time.monotonic()
    with (root / (label + '.log')).open('w') as log:
        log.write(json.dumps(command) + '\n'); log.flush()
        process = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=log, timeout=900)
    assert process.returncode == 0, (root / (label + '.log')).read_text()[-4000:]
    assert f'DiT backend: {a.backend}' in (root / (label + '.log')).read_text(), 'requested training device did not execute'
    if not (baseline and a.baseline_bin_dir):
        history = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        assert history and all(row['split'] == 'train' and row['epoch'] > 0 for row in history)
        assert all(math.isfinite(row[k]) for row in history for k in ('lr', 'loss', 'grad_norm', 'step_s'))
        assert all(row['grad_norm'] >= 0 and row['examples'] >= row['microbatches'] > 0 for row in history)
        assert len({row['session'] for row in history}) == 1
        assert [row['update'] for row in history] == list(range(history[0]['start_update'] + 1, history[-1]['update'] + 1))
        if label != 'resumed':
            assert history[0]['start_update'] == 0
        else:
            assert history[0]['start_update'] == 1
    results[label] = {'seconds': round(time.monotonic()-start, 3), 'command': command}
    return output

def generation(path):
    target = (path / (path / 'checkpoint.current').read_text().strip()).resolve()
    assert target.is_relative_to(path.resolve())
    return target

def same(left, right):
    for name in ('adapter_model.safetensors', 'adapter_config.json', 'optimizer_state.safetensors', 'trainer_state.json'):
        assert (generation(left) / name).read_bytes() == (generation(right) / name).read_bytes(), (left, right, name)

def log(label):
    return (root / (label + '.log')).read_text()

def hashes():
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in cache.glob('*.acecache')}

baseline = run('baseline', baseline=True)
cold = run('cold')
same(baseline, cold)
assert '0 reused, 3 written' in log('cold')
before = hashes()
warm = run('warm', ['--cache-memory-mb', '1'])
same(baseline, warm)
assert hashes() == before and '3 reused, 0 written' in log('warm')
for component in ('VAE-Enc', 'TextEncoder', 'CondEncoder'):
    assert f'[Load] {component}' not in log('warm'), 'warm cache loaded preprocessing models'
partial = run('partial', ['--steps', '1'])
partial_state = json.loads((generation(partial) / 'trainer_state.json').read_text())['continuation']
partial_metrics = json.loads((partial / 'metrics.jsonl').read_text().splitlines()[-1])
assert partial_metrics['microbatches'] == partial_metrics['examples'] == 2
assert math.isclose(partial_metrics['loss'], partial_state['epoch_loss'] / partial_state['epoch_batches'], rel_tol=1e-10)
resumed = run('resumed', ['--resume', partial])
same(baseline, resumed)
assert [row['microbatches'] for row in map(json.loads, (cold / 'metrics.jsonl').read_text().splitlines())] == [2, 1, 2, 1]
# Same-directory resume must append rather than erase the first invocation.
in_place = root / 'in-place'
shutil.copytree(partial, in_place)
in_place_command = results['partial']['command'][:-2]
in_place_command[in_place_command.index('--output') + 1] = str(in_place)
in_place_command += ['--resume', str(in_place)]
with (root / 'in-place-resume.log').open('w') as log_file:
    process = subprocess.run(in_place_command, cwd=root, env=env, stdout=log_file, stderr=log_file, timeout=900)
assert process.returncode == 0
same(baseline, in_place)
history = [json.loads(line) for line in (in_place / 'metrics.jsonl').read_text().splitlines()]
assert [row['update'] for row in history] == [1, 2, 3, 4]
assert len({row['session'] for row in history}) == 2
assert [row['start_update'] for row in history] == [0, 1, 1, 1]
if not a.quick:
    chunked = run('chunked', ['--preprocess-cache-dir', root / 'chunk-cache', '--preprocess-batch-size', '2'])
    same(baseline, chunked)
    damaged = sorted(cache.glob('*.acecache'))[0]
    payload = bytearray(damaged.read_bytes()); payload[-1] ^= 1; damaged.write_bytes(payload)
    repaired = run('repaired', ['--steps', '1'])
    same(partial, repaired)
    assert hashes() == before and '2 reused, 1 written, 1 corrupt/incompatible rebuilt' in log('repaired')
    # Change the longest caption so even reused entries need new dataset-wide padding.
    metadata = dataset / '2.txt'
    metadata.write_text(metadata.read_text().replace(captions[2], captions[2] + 'Bright plucked guitar. ' * 6))
    changed = run('changed-metadata', ['--steps', '1'])
    assert '2 reused, 1 written' in log('changed-metadata') and len(hashes()) == 4
    changed_reference = run('changed-metadata-reference', ['--steps', '1'], baseline=True)
    same(changed_reference, changed)
    # The three current entries are reusable when only optimizer settings change.
    run('changed-lr', ['--steps', '1', '--learning-rate', '0.0002'])
    assert '3 reused, 0 written' in log('changed-lr') and len(hashes()) == 4
    # Stop after a complete cache entry publishes, then reuse that partial cache.
    cancel_cache = root / 'cancel-cache'
    cancel_file = root / 'preprocess.cancel'
    status = root / 'preprocess-cancel.json'
    cancelled_output = root / 'preprocess-cancelled'
    command = [str(binary / ('ace-train' + suffix))] + common + ['--output', str(cancelled_output),
        '--preprocess-cache-dir', str(cancel_cache), '--cancel-file', str(cancel_file), '--progress-file', str(status)]
    with (root / 'preprocess-cancel.log').open('w') as output:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=output, stderr=output)
        try:
            deadline = time.monotonic() + 900
            while process.poll() is None:
                assert time.monotonic() < deadline
                if list(cancel_cache.glob('*.acecache')):
                    cancel_file.write_text('stop after a complete cache entry')
                time.sleep(.01)
            assert process.returncode == 0 and json.loads(status.read_text())['status'] == 'cancelled'
            assert not (cancelled_output / 'checkpoint.current').exists()
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=30)
    cached_count = len(list(cancel_cache.glob('*.acecache')))
    assert 0 < cached_count < 3
    continued = run('preprocess-continued', ['--preprocess-cache-dir', cancel_cache, '--steps', '1'])
    changed_current_reference = run('preprocess-reference', ['--steps', '1'], baseline=True)
    same(changed_current_reference, continued)
    assert f'{cached_count} reused, {3-cached_count} written' in log('preprocess-continued')
results.update({'passed': True, 'backend': a.backend, 'device': device['description'],
    'adapter_type': a.adapter_type, 'quick': a.quick, 'variable_conditioning_lengths': True,
    'cold_warm_uncached_bitexact': True, 'mid_epoch_resume_bitexact': True})
(root / 'summary.json').write_text(json.dumps(results, indent=2))
print('PASS cache cold/warm/uncached parity, streaming resume and invalidation', flush=True)
