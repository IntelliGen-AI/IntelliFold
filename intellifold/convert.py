#!/usr/bin/env python3
"""STAGE 2 of 2 -- convert IntelliFold-v2 PyTorch weights to AF3 haiku `.bin.zst`.

Uses ONLY the schema artifact from stage 1 (`debug/extract_af3_schema.py`) plus the
IntelliFold-v2 checkpoint -- **no gated AlphaFold 3 weights are needed here**. The
schema carries the af3.bin record templates (dtype/shape), the small __meta__
records, and the PT<->haiku correspondence; this stage applies that map to v2 (with
v2's "full_fat" head counts) and writes the loadable weights + v2's Fourier embedding.

This module is SELF-CONTAINED (the AF3 .bin codec + the `convert()` builder + the .pt
loader live here). The stage-1 schema extractor (`debug/extract_af3_schema.py`) imports
those from here, so production stage-2 never depends on the dev-time stage-1 tool.

Pipeline:
  1. load af3_schema.pkl                          -> record templates + meta + correspondence
  2. load intellifold_v2.pt                       -> {name: float32 ndarray}
  3. apply the map (V2_HEADS) to the schema       -> intellifold_v2.bin.zst (full_fat dims)
  4. extract v2's trained Fourier noise-embedding -> intfold_fourier.npz (needed at RUN time)

Usage
-----
  python convert_ifv2_to_jax.py \
      --schema  af3_schema.pkl \
      --v2-pt   intellifold_v2.pt \
      --out-dir ./out

Outputs (in --out-dir):
  intellifold_v2.bin.zst   the converted weights -> point AF3 --model_dir at this dir
  intfold_fourier.npz      v2's Fourier weight[256]/bias[256] -> inject at runtime (see README)

Requirements: numpy, zstandard, ml_dtypes, torch (torch only to load the .pt).
Runs on CPU, no GPU needed. ~2-3 GB RAM.

NOTE: converts the advanced_init=False path (no LinearCat). IntelliFold-v2 is
architecturally identical to AF3 except for the hidden sizes (full_fat, see V2_HEADS).
"""
import argparse, io, struct, os, pickle
import numpy as np
import ml_dtypes        # noqa: F401  (registers the bfloat16 dtype used by af3.bin)
import zstandard

# IntelliFold-v2 "full_fat" preset (the conversion target): c_z=512/no_heads_pair=8,
# c_m=256, c_t=256/no_heads_template=8 differ from the public af3.bin; everything else
# (c_s=384/16, c_token=768/16, c_atom=128/4) is identical.
V2_HEADS = dict(pair=8, template=8, msa=8, single=16, token=16, atom=4)

# ---- structural special case: pair-bias linear_z folds into pair_logits_projection ----
ATOM_PL = {
    "diffuser/evoformer_conditioning_atom_transformer_encoder/pair_logits_projection":
        "backbone_trunk.input_embedder.atom_attention_encoder.atom_transformer",
    "diffuser/~/diffusion_head/diffusion_atom_transformer_encoder/pair_logits_projection":
        "diffusion_module.atom_attention_encoder.atom_transformer",
    "diffuser/~/diffusion_head/diffusion_atom_transformer_decoder/pair_logits_projection":
        "diffusion_module.atom_attention_decoder.atom_transformer",
}
DIFFTX_PL     = "diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer/pair_logits_projection"
DIFFTX_PREFIX = "diffusion_module.diffusion_transformer"


# ============================================================================
# AF3 .bin codec  (record format verbatim from the official alphafold3 params.py)
# ============================================================================
def encode_record(scope, name, arr):
    scope = scope.encode("utf-8"); name = name.encode("utf-8")
    shape = arr.shape; dtype = str(arr.dtype).encode("utf-8")
    arr = np.ascontiguousarray(arr); arr_buffer = arr.tobytes("C")
    header = struct.pack("<5i", len(scope), len(name), len(dtype), len(shape), len(arr_buffer))
    return header + b"".join((scope, name, dtype, struct.pack(f"{len(shape)}i", *shape), arr_buffer))


def _read_record(stream):
    hs = struct.calcsize("<5i"); header = stream.read(hs)
    if not header: return None
    (sl, nl, dl, shl, bl) = struct.unpack("<5i", header)
    fmt = f"<{sl}s{nl}s{dl}s{shl}i"; payload = stream.read(struct.calcsize(fmt) + bl)
    scope, name, dtype, *shape = struct.unpack_from(fmt, payload)
    arr = np.frombuffer(payload[-bl:], dtype=dtype.decode()).reshape(shape)
    return scope.decode(), name.decode(), arr


def read_bin(path):
    with open(path, "rb") as fh:
        data = zstandard.ZstdDecompressor().stream_reader(fh).read() if path.endswith(".zst") else fh.read()
    buf = io.BytesIO(data); out = []
    while True:
        r = _read_record(buf)
        if r is None: break
        out.append(r)
    return out


# ============================================================================
# Apply the correspondence (schema-driven, per record)
# ============================================================================
def classify(scope):
    if "template_embedding" in scope and "pair_attention" in scope: return "template"
    if "single_attention" in scope: return "single"
    if "pair_attention"   in scope: return "pair"
    if "msa_attention"    in scope: return "msa"
    if "diffusion_head/transformer" in scope: return "token"
    if "atom_transformer" in scope: return "atom"
    return None


def per_block_target(pt, tag, cls, scope, af3_per, heads_cfg):
    if tag == "id": return tuple(pt.shape)
    if tag == "T":  return (pt.shape[1], pt.shape[0])
    if tag == "reshape":
        if cls in heads_cfg:
            H = heads_cfg[cls]
            return (H, pt.shape[0] // H, pt.shape[1]) if pt.ndim == 2 else (H, pt.shape[0] // H)
        assert "template_pair_embedding" in scope, scope
        return (int(np.prod(pt.shape)),)
    if tag == "T_reshape":
        if cls in heads_cfg:
            H = heads_cfg[cls]
            return (pt.shape[1], H, pt.shape[0] // H)
        return (pt.shape[1],) + tuple(af3_per[1:])
    raise ValueError(tag)


def apply_tag(pt, tag, target):
    if   tag == "id":        out = pt
    elif tag == "T":         out = pt.T
    elif tag == "reshape":   out = pt.reshape(target)
    elif tag == "T_reshape": out = np.ascontiguousarray(pt.T).reshape(target)
    else: raise ValueError(tag)
    assert tuple(out.shape) == tuple(target), f"{tag}: {out.shape}!={target}"
    return out


def build_pair_logits(scope, arr, pt):
    """Pack N per-block linear_z (transposed) into the pair_logits_projection layer axis."""
    if scope in ATOM_PL:
        cap, nb, hd = arr.shape
        z0 = pt[f"{ATOM_PL[scope]}.blocks.0.attention_pair_bias.linear_z.weight"]
        out = np.zeros((z0.shape[1], nb, z0.shape[0]), np.float32)
        for j in range(nb):
            out[:, j, :] = pt[f"{ATOM_PL[scope]}.blocks.{j}.attention_pair_bias.linear_z.weight"].T
        return out
    if scope == DIFFTX_PL:                       # diffusion transformer = nested 6x4 stack
        G, _, INNER, _ = arr.shape
        z0 = pt[f"{DIFFTX_PREFIX}.blocks.0.attention_pair_bias.linear_z.weight"]
        out = np.zeros((G, z0.shape[1], INNER, z0.shape[0]), np.float32)
        for g in range(G):
            for j in range(INNER):
                out[g, :, j, :] = pt[f"{DIFFTX_PREFIX}.blocks.{g*INNER+j}.attention_pair_bias.linear_z.weight"].T
        return out
    return None


def convert(pt, heads_cfg, out_path, recs, amap, compare):
    """Walk the af3.bin schema; for each record build the value from PT and write it.
    Unused AF3 params and missing v2 keys (e.g. v2's dead 4th-MSA-block) are zero-filled.
    With compare=True, every used record must match the af3.bin template bit-for-bit
    (recs must then carry the real af3 values). Shared by stage 1 (self-check) and
    stage 2 (v2 conversion, where recs carry zero placeholders for non-meta records)."""
    af3 = {(s, n): a for s, n, a in recs}
    nz = ub = cmp_ok = cmp_tot = 0; mism = []
    with zstandard.ZstdCompressor(level=10).stream_writer(open(out_path, "wb")) as comp:
        for scope, name, arr in recs:
            if scope.startswith("__") or name.startswith("__"):
                built = np.asarray(arr); kind = "meta"            # carry meta record verbatim
            elif name == "weights" and (scope in ATOM_PL or scope == DIFFTX_PL):
                built = build_pair_logits(scope, arr, pt).astype(arr.dtype); kind = "used"
            else:
                nsd = scope.count("__layer_stack"); cls = classify(scope)
                if nsd == 0:
                    hit = amap.get((scope, name, ()))
                    if hit is None or hit[0] not in pt:
                        built = np.zeros(arr.shape, arr.dtype); ub += 1; kind = "unused"
                    else:
                        pk, tag = hit
                        built = apply_tag(pt[pk], tag,
                                          per_block_target(pt[pk], tag, cls, scope, arr.shape, heads_cfg)).astype(arr.dtype)
                        kind = "used"
                else:                                              # stacked block -> fill each layer slot
                    per = None; slots = {}
                    for idx in np.ndindex(*arr.shape[:nsd]):
                        si = tuple(int(i) for i in idx); hit = amap.get((scope, name, si)); slots[si] = hit
                        if per is None and hit and hit[0] in pt:
                            per = per_block_target(pt[hit[0]], hit[1], cls, scope, arr.shape[nsd:], heads_cfg)
                    assert per is not None, f"{scope}::{name}"
                    o = np.zeros(tuple(arr.shape[:nsd]) + tuple(per), np.float32)
                    for si, hit in slots.items():
                        if hit and hit[0] in pt: o[si] = apply_tag(pt[hit[0]], hit[1], per)
                        else: nz += 1                              # missing slot (e.g. v2 dead MSA block) -> zero
                    built = o.astype(arr.dtype); kind = "used"
            if kind == "used" and compare:
                cmp_tot += 1; a = np.ascontiguousarray(af3[(scope, name)]); b = np.ascontiguousarray(built)
                if a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(): cmp_ok += 1
                else: mism.append((scope, name))
            comp.write(encode_record(scope, name, np.ascontiguousarray(built)))
    print(f"[{out_path}]  records={len(recs)}  unused_zerofill={ub}  zerofill_slices={nz}", flush=True)
    if compare:
        print(f"[self-check]  used records bit-exact vs af3.bin: {cmp_ok}/{cmp_tot}  mismatches={len(mism)}", flush=True)
        for s, n in mism[:10]: print("   MISMATCH", s, n, flush=True)
        if mism:
            raise SystemExit("self-check FAILED -- correspondence derivation is wrong, aborting.")


def load_pt(path):
    """Load a .pt checkpoint -> {name: float32 ndarray}.  (torch imported lazily.)"""
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.detach().cpu().float().numpy() for k, v in sd.items()}


# ============================================================================
# Stage 2: schema -> v2 weights
# ============================================================================
def load_schema(path):
    """Rebuild (recs, amap) from the stage-1 schema. Non-meta records get zero
    placeholders of the right dtype/shape (convert() only reads their shape/dtype);
    __meta__ records carry their real values."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    if d.get("version") != 1:
        raise SystemExit(f"unsupported schema version: {d.get('version')!r}")
    recs = []
    for i, rm in enumerate(d["records"]):
        if rm["meta"]:
            arr = d["meta_arrays"][str(i)]
        else:
            arr = np.zeros(tuple(rm["shape"]), dtype=np.dtype(rm["dtype"]))
        recs.append((rm["scope"], rm["name"], arr))
    amap = {(c["scope"], c["name"], tuple(c["sidx"])): (c["pt_key"], c["tag"]) for c in d["corr"]}
    return recs, amap


def extract_fourier(v2_pt, out_path):
    """Save v2's trained Fourier noise-embedding (AF3 hardcodes its own; v2 differs)."""
    w = b = None
    for k, v in v2_pt.items():
        if k.endswith("fourier_embedding.weight"): w = np.ravel(v).astype("float32")
        if k.endswith("fourier_embedding.bias"):   b = np.ravel(v).astype("float32")
    assert w is not None and b is not None, "fourier_embedding.{weight,bias} not found in v2 checkpoint"
    np.savez(out_path, weight=w, bias=b)
    print(f"[{out_path}]  fourier weight{w.shape} bias{b.shape}  weight[:3]={w[:3]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema",  required=True, help="schema artifact from debug/extract_af3_schema.py (af3_schema.pkl)")
    ap.add_argument("--v2-pt",   required=True, help="IntelliFold-v2 PyTorch weights (advanced_init=False)")
    ap.add_argument("--out-dir", default=".",   help="output directory")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    v2_out  = os.path.join(a.out_dir, "intellifold_v2.bin.zst")
    fourier = os.path.join(a.out_dir, "intfold_fourier.npz")

    print(f"loading schema {a.schema} ...", flush=True)
    recs, amap = load_schema(a.schema)
    print(f"  {len(recs)} records, {len(amap)} correspondences", flush=True)

    print("loading IntelliFold-v2 PyTorch weights ...", flush=True)
    v2_pt = load_pt(a.v2_pt); print(f"  {len(v2_pt)} tensors", flush=True)

    convert(v2_pt, V2_HEADS, v2_out, recs, amap, compare=False)
    extract_fourier(v2_pt, fourier)

    print("\nDONE. Point --model-dir at the directory containing intellifold_v2.bin.zst")
    print("(renamed to af3.bin.zst), or just use `intellifold predict`, which downloads,")
    print("converts and applies the two runtime patches for you.")


if __name__ == "__main__":
    main()
