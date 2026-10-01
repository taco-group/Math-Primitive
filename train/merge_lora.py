"""Merge an Absorb LoRA checkpoint into the ORIGINAL Qwen3.5 checkpoint for vLLM serving.

    python merge_lora.py <base_snapshot_dir> <checkpoint_dir> <out_dir>

Training uses the text-only Qwen3_5ForCausalLM view of the model, so adapter keys look like
`base_model.model.model.layers.N.<module>.lora_{A,B}.weight`. The released Qwen3.5 checkpoints
are composite `Qwen3_5ForConditionalGeneration` models (vision tower + language tower) whose
language-tower keys are `model.language_model.layers.N.<module>.weight`; vLLM serves only this
composite layout. This script therefore writes the base shards back out with

    W' = bf16( W + (lora_alpha / r) * B @ A )      (computed in fp32, rounded once)

applied to the language-tower targets, leaves every other tensor (incl. the vision tower)
untouched, and copies the config / tokenizer / generation-config files unchanged. Runs on CPU.

The merged LoRA targets are bit-identical to PEFT's merge_and_unload(). One difference from
"load in bf16 + merge_and_unload + graft": that route also rounds the fp32 linear-attention
norm weights (`linear_attn.norm.weight`) to bf16, while this script keeps them exact. The
released 9B was merged with this script; the released 4B/27B went through the PEFT route
(matching the checkpoints that were evaluated), so their norm weights are bf16-rounded.
"""
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    base, ckpt, out = map(Path, sys.argv[1:])
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(ckpt / "adapter_config.json"))
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), "only plain LoRA is supported"
    scale = cfg["lora_alpha"] / cfg["r"]

    index = json.load(open(base / "model.safetensors.index.json"))
    wmap = index["weight_map"]

    adapter = load_file(str(ckpt / "adapter_model.safetensors"))
    deltas = {}
    for k in adapter:
        if ".lora_A." not in k:
            continue
        name = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
        if name not in wmap:  # composite checkpoint: the language tower lives under model.language_model.
            name = name.replace("model.layers.", "model.language_model.layers.", 1)
        assert name in wmap, f"adapter target {name} not found in the base checkpoint"
        deltas[name] = scale * (adapter[k.replace(".lora_A.", ".lora_B.")].float() @ adapter[k].float())
    print(f"{len(deltas)} LoRA targets, scale={scale}")

    applied = 0
    for shard in sorted(set(wmap.values())):
        tensors = {}
        with safe_open(str(base / shard), framework="pt") as f:
            meta = f.metadata()
            for name in f.keys():
                t = f.get_tensor(name)
                if name in deltas:
                    assert deltas[name].shape == t.shape, name
                    t = (t.float() + deltas[name]).to(t.dtype)
                    applied += 1
                tensors[name] = t
        save_file(tensors, str(out / shard), metadata=meta or {"format": "pt"})
        print(f"wrote {shard}")
    assert applied == len(deltas), (applied, len(deltas))

    shards = set(wmap.values())
    for f in base.iterdir():
        if f.is_file() and f.name not in shards and f.name != "README.md":
            shutil.copy(f, out / f.name)
    print(f"done: {applied} tensors merged -> {out}")


if __name__ == "__main__":
    main()
