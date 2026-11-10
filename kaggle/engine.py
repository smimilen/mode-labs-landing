"""FASHN engine copied from the completed Modegen v1.1 GPU probe. No training or private assets."""

import os
from pathlib import Path

TRANSFORMERS_BACKENDS = {
    'USE_TORCH': '1', 'USE_TF': '0', 'USE_FLAX': '0',
    'FORCE_TF_AVAILABLE': '0',
}

os.environ.update(TRANSFORMERS_BACKENDS)

SOURCES = {
    'fashn-vton-1.5': ('https://github.com/fashn-AI/fashn-vton-1.5.git',
                     '7c0f10af3f91ad4048fe9729c470a13ef905d25a'),
    'fashn-human-parser': ('https://github.com/fashn-AI/fashn-human-parser.git',
                         'f2771f2fb8655349e87e2869bbde7ace0bd06f2c'),
}

WEIGHTS = [
    ('fashn-ai/fashn-vton-1.5', '7720683168567eb5a2a4c67f15116c6e29c83ded',
     '', ['model.safetensors']),
    ('fashn-ai/DWPose', '548b5df25b84d9f4aac0611dfa1c2a7a12f15571',
     'dwpose', ['yolox_l.onnx', 'dw-ll_ucoco_384.onnx']),
    ('fashn-ai/fashn-human-parser', '1f80c34dbab321c5730dda5c3fea279fd3e97498',
     'human_parser', ['config.json', 'model.safetensors']),
]

def memory_estimate(query_chunk=512):
    """Exact storage/score sizes, NOT a measured or guaranteed total VRAM peak.

    Source: pinned TryOnModel defaults; independently matches HF metadata.
    971,813,808 parameters + 432 unused checkpoint-compatible buffer values.
    CFG doubles batch: 2 x 10 heads x 6912 tokens (two 72x48 images).
    """
    h = 1280
    linear = lambda a, b: a * b + b
    single = linear(h, 7*h) + linear(5*h, h) + 256 + linear(h, 3*h)
    stream = (linear(h, 6*h) + linear(h, 3*h) + 256 + linear(h, h)
              + linear(h, 4*h) + linear(4*h, h))
    count = (linear(256, h) + linear(h, h) + 4*h + 11*h*144 + 2*h
             + 20*single + 16*stream + linear(h, 432) + linear(h, 2*h) + 432)
    assert count == 971814240
    tokens = 2 * (864 // 12) * (576 // 12)
    queries = min(query_chunk, tokens) if query_chunk else tokens
    return dict(parameters=count-432, buffer_values=432, stored_values=count,
                fp32_model_bytes=count*4, cfg_batch=2, sequence_tokens=tokens,
                full_score_bytes=2*10*tokens*tokens*4,
                chunk_score_bytes=2*10*queries*tokens*4,
                note='Storage and one score tensor only; other live tensors/workspaces are additional.')

def download_weights(out):
    from huggingface_hub import hf_hub_download
    root = out / 'weights'
    for repo, revision, folder, names in WEIGHTS:
        for name in names:
            hf_hub_download(repo_id=repo, revision=revision, filename=name,
                            local_dir=str(root / folder), token=False)
    return root

def make_pipeline(weights, preprocess_device, query_chunk):
    import torch
    from torch.nn import functional as F
    from safetensors.torch import load_file
    from fashn_vton import TryOnPipeline
    from fashn_vton.dwpose import DWposeDetector
    from fashn_human_parser import FashnHumanParser
    from fashn_vton import tryon_mmdit

    # Each query still attends to ALL keys. No mask/CFG/precision change.
    # Query slicing bounds math-SDPA score storage even if fused FP32 is unavailable.
    if query_chunk:
        def chunked_attention(q, k, v):
            return torch.cat([
                F.scaled_dot_product_attention(q[:, :, i:i+query_chunk], k, v)
                for i in range(0, q.shape[2], query_chunk)
            ], dim=2)
        tryon_mmdit._attn_processor = chunked_attention
    else:
        tryon_mmdit._attn_processor = F.scaled_dot_product_attention

    class ProbePipeline(TryOnPipeline):
        def _setup_tryon_model(self):
            self.inference_dtype = torch.float32  # Explicit even on a newer GPU.
            self.tryon_model = tryon_mmdit.TryOnModel()
            state = load_file(str(weights / 'model.safetensors'), device='cpu')
            self.tryon_model.load_state_dict(state, strict=True)
            del state  # Avoid staging the BF16 checkpoint on cuda:0 as well.
            self.tryon_model.to(self.device, dtype=torch.float32).eval().requires_grad_(False)

        def _setup_pose_model(self):
            self.pose_model = DWposeDetector(
                checkpoints_dir=str(weights / 'dwpose'), device=preprocess_device)

        def _setup_hp_model(self):
            # model_id accepts a local from_pretrained directory; no implicit main download.
            self.hp_model = FashnHumanParser(
                model_id=str(weights / 'human_parser'), device=preprocess_device)

    pipe = ProbePipeline(weights_dir=str(weights), device='cuda:0')
    tensors = list(pipe.tryon_model.parameters()) + list(pipe.tryon_model.buffers())
    if sum(t.numel() for t in tensors) != memory_estimate()['stored_values']:
        raise RuntimeError('Generator shape/count differs from the pinned model.')
    if any(t.dtype != torch.float32 or str(t.device) != 'cuda:0' for t in tensors):
        raise RuntimeError('Generator must be entirely FP32 on cuda:0.')
    if str(next(pipe.hp_model.model.parameters()).device) != preprocess_device:
        raise RuntimeError('Human parser device mismatch.')
    provider = 'CPUExecutionProvider' if preprocess_device == 'cpu' else 'CUDAExecutionProvider'
    wholebody = pipe.pose_model.pose_estimation
    for session in (wholebody.session_det, wholebody.session_pose):
        if session.get_providers()[0] != provider:
            raise RuntimeError('Pose provider fell back; rerun with CPU preprocessing.')
        if preprocess_device == 'cuda:1':
            if session.get_provider_options()[provider].get('device_id') != '1':
                raise RuntimeError('Pose CUDA provider is not on device 1.')
    return pipe
