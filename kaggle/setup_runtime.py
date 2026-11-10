"""Future Kaggle setup: preserve installed Torch/CUDA, fetch code only, no weights."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

from engine import SOURCES, TRANSFORMERS_BACKENDS
DEFAULT_OUT = "/kaggle/working/modegen_web"

IMPORT_PREFLIGHT = '''
import sys
import torch, torchvision, onnxruntime
from transformers.utils import is_torch_available, is_tf_available, is_flax_available
assert is_torch_available(), 'PyTorch backend must be available.'
assert not is_tf_available() and not is_flax_available(), 'Optional backends were not disabled.'
from fashn_vton import TryOnPipeline
from fashn_human_parser import FashnHumanParser
unexpected = sorted(n for n in sys.modules if n.split('.')[0] in ('tensorflow', 'jax', 'flax'))
assert not unexpected, 'Unexpected optional backend import: ' + repr(unexpected[:5])
print('Import-only preflight passed: PyTorch only; TF/JAX/Flax not imported; no weights loaded.')
'''


def run(args):
    subprocess.run(args, check=True)


def protected_runtime():
    return {d.metadata['Name']: d.version for d in metadata.distributions()
            if d.metadata['Name'].lower().replace('_', '-').startswith(('torch', 'nvidia-'))
            or d.metadata['Name'].lower() == 'triton'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT_OUT)
    parser.add_argument('--preprocess-device', choices=['cpu', 'cuda:1'], default='cpu')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from packaging.version import Version
    if sys.version_info < (3, 10) or Version(torch.__version__.split('+')[0]) < Version('2.2'):
        raise RuntimeError('Requires existing Python >=3.10 and Torch >=2.2 (parser requirement). No Torch replacement.')
    before = protected_runtime()
    metadata.version('torchvision')  # Fail before pip if the preinstalled pair is incomplete.
    if args.preprocess_device == 'cuda:1':
        if (not (torch.version.cuda or '').startswith('12.')
                or (torch.backends.cudnn.version() or 0) < 90000):
            raise RuntimeError('ORT 1.20.1 GPU path needs CUDA 12/cuDNN 9; use CPU preprocessing.')
        if torch.cuda.device_count() < 2:
            raise RuntimeError('cuda:1 needs two visible GPUs; use CPU preprocessing.')
    constraint = out / 'preserve_runtime.txt'
    constraint.write_text(''.join(f'{name}=={version}\n' for name, version in sorted(before.items())))
    ort = 'onnxruntime' if args.preprocess_device == 'cpu' else 'onnxruntime-gpu'
    other = 'onnxruntime-gpu' if ort == 'onnxruntime' else 'onnxruntime'
    try:
        metadata.version(other)
    except metadata.PackageNotFoundError:
        pass
    else:
        run([sys.executable, '-m', 'pip', 'uninstall', '-y', other])
        # Both ORT distributions share files: reinstall the selected one after removal.
        run([sys.executable, '-m', 'pip', 'uninstall', '-y', ort])
    deps = ['numpy>=1.26,<2', 'transformers==4.46.3', 'huggingface_hub>=0.23,<1',
            'gradio==5.49.1', 'safetensors>=0.4', 'Pillow>=9', 'einops>=0.6', 'tqdm>=4.65',
            'opencv-python==4.10.0.84', 'matplotlib>=3.5', f'{ort}==1.20.1']
    run([sys.executable, '-m', 'pip', 'install', '--quiet', '--constraint', str(constraint),
         '--upgrade-strategy', 'only-if-needed', *deps])
    for name, (url, revision) in SOURCES.items():
        repo = out / 'source' / name
        if not repo.exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            run(['git', 'clone', '--no-checkout', url, str(repo)])
        run(['git', '-C', str(repo), 'checkout', '--detach', revision])
        actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise RuntimeError('Source pin mismatch: ' + name)
        # Explicit deps above; the upstream VTON package requests onnxruntime-gpu even on CPU.
        run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps', str(repo)])
    if protected_runtime() != before:
        raise RuntimeError('Torch/CUDA package versions changed unexpectedly.')
    # Fresh process avoids stale imports after pip; constructors/weights are never invoked.
    run([sys.executable, '-c', IMPORT_PREFLIGHT])
    (out / 'versions.json').write_text(json.dumps(dict(
        python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        runtime_preserved=before, sources=SOURCES, preprocess_device=args.preprocess_device,
        transformers_backends=TRANSFORMERS_BACKENDS), indent=2))
    with (out / 'pip_freeze.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    print('Setup complete; existing Torch/CUDA preserved. No weights downloaded.', flush=True)


if __name__ == '__main__':
    main()
