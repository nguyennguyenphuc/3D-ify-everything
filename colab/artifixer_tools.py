"""Small entry points run inside the ArtiFixer environment (cwd = vendor/ArtiFixer).

    python colab/artifixer_tools.py caption --out caption.h5 --text "..." --images DIR
    python colab/artifixer_tools.py infer <model_eval.run_inference args>
    python colab/artifixer_tools.py export --checkpoint ckpt.pt --out model.ply
"""
import argparse
import importlib.util
import runpy
import sys
from pathlib import Path


def caption(argv):
    """Write the prompt HDF5 ArtiFixer reads, from a fixed text caption.

    ArtiFixer's own captioner loads Qwen3-VL-30B-A3B; inference only needs the
    umt5 embedding of *a* caption, so encode a supplied text instead.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--text', required=True)
    p.add_argument('--model_id', default='Wan-AI/Wan2.1-T2V-1.3B-Diffusers')
    p.add_argument('--images', type=Path, required=True)
    a = p.parse_args(argv)
    import h5py
    import numpy as np
    from data_processing.captioning.generate_captions import generate_text_embedding, get_text_encoder_and_tokenizer
    names = sorted(x.name for x in a.images.iterdir() if x.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    encoder, tokenizer = get_text_encoder_and_tokenizer(a.model_id)
    embedding = generate_text_embedding(a.text, encoder, tokenizer, 512)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix('.tmp')
    with h5py.File(tmp, 'w') as f:
        d = f.create_dataset(names[0], data=embedding)
        d.attrs['caption'] = a.text
        d.attrs['image_indices'] = np.arange(len(names))
    tmp.rename(a.out)
    print(f'caption embedding {embedding.shape} -> {a.out}', flush=True)


def infer(argv):
    """model_eval.run_inference with an SDPA fallback when FA3 is not installed.

    ArtiFixer selects the ``_flash_3`` dispatch backend on any sm_90 GPU; Colab
    H100 runtimes do not ship the FA3 Hopper build, so use diffusers' auto SDPA.
    """
    if importlib.util.find_spec('flash_attn_interface') is None:
        import model_training.net.transformer as transformer
        if transformer._DEFAULT_ATTENTION_BACKEND == '_flash_3':
            transformer._DEFAULT_ATTENTION_BACKEND = None
            print('FA3 không có: dùng auto SDPA cho dispatch attention', flush=True)
    sys.argv = ['model_eval.run_inference', *argv]
    runpy.run_module('model_eval.run_inference', run_name='__main__', alter_sys=True)


def prep(argv):
    """data_processing.prepare_colmap_artifixer_inputs with a non-empty 3DGUT val split.

    ArtiFixer selects every source view for training by default, and 3DGRUT's
    val split is "all views not selected", so it is empty and
    compute_spatial_extents crashes. When the selection covers every view,
    validate on all views instead (test_split_interval=0 means no hold-out).
    """
    import json
    import struct
    from threedgrut.datasets.dataset_colmap import ColmapDataset
    original = ColmapDataset.__init__

    def init(self, path, *args, split='train', selected_indices_file=None, **kwargs):
        if split == 'val' and selected_indices_file and not kwargs.get('image_path_override'):
            with open(Path(path)/'sparse'/'0'/'images.bin', 'rb') as f: frames = struct.unpack('<Q', f.read(8))[0]
            limit = kwargs.get('num_selected_indices') or frames
            if len(set(json.load(open(selected_indices_file))[:limit])) >= frames:
                print('Mọi ảnh đều dùng để train: val 3DGUT dùng lại toàn bộ ảnh', flush=True)
                selected_indices_file, kwargs['test_split_interval'] = None, 0
        original(self, path, *args, split=split, selected_indices_file=selected_indices_file, **kwargs)

    ColmapDataset.__init__ = init
    sys.argv = ['data_processing.prepare_colmap_artifixer_inputs', *argv]
    runpy.run_module('data_processing.prepare_colmap_artifixer_inputs', run_name='__main__', alter_sys=True)


def export(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args(argv)
    import torch
    from threedgrut.model.model import MixtureOfGaussians
    checkpoint = torch.load(a.checkpoint, weights_only=False)
    model = MixtureOfGaussians(checkpoint['config'])
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    model.export_ply(str(a.out))
    print(f'PLY -> {a.out} ({model.get_positions().shape[0]} Gaussian)', flush=True)


if __name__ == '__main__':
    command, rest = sys.argv[1], sys.argv[2:]
    {'caption': caption, 'prep': prep, 'infer': infer, 'export': export}[command](rest)
