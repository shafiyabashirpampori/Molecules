"""Predict top-25 SMILES per test molecule → submission.csv.

Fixes applied (v2):
  1. meta[mid] bug: previously only the LAST spectrum's precursor_mz and adduct
     were kept. Now all spectra per molecule are accumulated; majority-vote adduct
     and mean precursor_mz are used for the mass filter.
  2. Logit-space fusion: raw logits from all spectra are averaged BEFORE sigmoid,
     so high-energy and low-energy spectra are fused in an unconstrained space
     rather than squashing each prediction through sigmoid first.
  3. Fallback adduct handling: unknown adducts default to [M+H]+ instead of
     silently returning None neutral mass → triggering an unfiltered global search.
"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict, Counter
from pathlib import Path

from src.config import pick_device, CFG
from src.model import SpectrumFPNet
from src.retrieval import retrieve_topk
from src.utils import get_logger, set_seed

log = get_logger("predict")


# ---------------------------------------------------------------------------
# Fusion helpers
# ---------------------------------------------------------------------------
def fuse_logits(raw_logits: list) -> np.ndarray:
    """Average raw model logits across all spectra, THEN apply sigmoid.

    This is strictly better than averaging sigmoid outputs because:
    - In logit space, the contributions are unconstrained and additive.
    - Applying sigmoid once after averaging correctly reflects the geometric
      mean of the individual posterior probabilities.
    """
    stacked = np.stack(raw_logits, axis=0)   # (n_spectra, 2048)
    mean_logit = stacked.mean(axis=0)         # (2048,)
    return 1.0 / (1.0 + np.exp(-mean_logit)) # sigmoid → probabilities


def select_adduct(adduct_list: list) -> str:
    """Return majority-vote adduct string; fall back to [M+H]+ if unknown."""
    if not adduct_list:
        return "[M+H]+"
    counter = Counter(adduct_list)
    top = counter.most_common(1)[0][0]
    # Validate it is in our known adduct table
    if top not in CFG.ADDUCT_DELTA:
        # Try the most frequent known adduct
        for adduct, _ in counter.most_common():
            if adduct in CFG.ADDUCT_DELTA:
                return adduct
        return "[M+H]+"
    return top


def select_precursor_mz(mz_list: list) -> float:
    """Return mean precursor m/z across all spectra for this molecule."""
    return float(np.mean(mz_list))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed()
    device = pick_device()

    # ── Load model ──────────────────────────────────────────────────────────
    ckpt = torch.load(
        CFG.MODEL_DIR / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    model = SpectrumFPNet(ckpt["in_dim"], CFG.FP_BITS).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(f"Loaded SpectrumFPNet  AP={ckpt.get('ap', 0):.4f}  "
             f"epoch={ckpt.get('epoch', '?')}  device={device}")

    # ── Accumulate per-molecule logits + metadata ───────────────────────────
    # FIX 1 + 2: store raw logits (not sigmoid outputs) and ALL adducts / mzs
    mol_logits   = defaultdict(list)   # mid → [logit_array, ...]
    mol_adducts  = defaultdict(list)   # mid → [adduct_str, ...]
    mol_pmzs     = defaultdict(list)   # mid → [precursor_mz, ...]

    chunk_dir = CFG.PROC_DIR / "test_chunks"
    test_files = sorted(chunk_dir.glob("test_*.npz"))
    if not test_files:
        fallback = CFG.PROC_DIR / "test.npz"
        if fallback.is_file():
            test_files = [fallback]
        else:
            raise FileNotFoundError(
                f"No processed test data found in {chunk_dir}. "
                "Run: python -m src.preprocessing"
            )

    for test_file in test_files:
        log.info(f"Inferring on {test_file.name}")
        with np.load(test_file, allow_pickle=True) as te:
            spectra  = te["spectra"].astype(np.float32, copy=False)
            pmzs     = te["precursor_mz"]
            adducts  = te["adduct"]
            mol_ids  = te["molecule_id"]

            # Append normalised precursor_mz as 12001st feature (same as training)
            X = np.concatenate(
                [spectra, (pmzs / 1000.0).reshape(-1, 1).astype(np.float32)],
                axis=1,
            )

            # Batch inference – collect RAW LOGITS (before sigmoid)
            raw_preds = []
            with torch.no_grad():
                for i in tqdm(range(0, len(X), 256), desc="infer"):
                    xb = torch.from_numpy(X[i : i + 256]).to(device)
                    raw_preds.append(model(xb).cpu().numpy())   # logits, not sigmoid
            raw_preds = np.concatenate(raw_preds, axis=0)       # (N, 2048)

        for i, mid in enumerate(mol_ids):
            mol_logits[mid].append(raw_preds[i])
            mol_adducts[mid].append(str(adducts[i]))
            mol_pmzs[mid].append(float(pmzs[i]))

    log.info(f"Collected logits for {len(mol_logits)} unique molecules.")

    # ── Retrieve top-25 candidates per molecule ─────────────────────────────
    rows = []
    for mid, logit_list in tqdm(mol_logits.items(), desc="retrieve"):
        # FIX 2: fuse in logit space → single sigmoid at the end
        fp = fuse_logits(logit_list)

        # FIX 1: use majority adduct + mean precursor mass
        adduct = select_adduct(mol_adducts[mid])
        pmz    = select_precursor_mz(mol_pmzs[mid])

        smis, _, _ = retrieve_topk(fp, pmz, adduct, top_k=CFG.TOP_K)

        # Guarantee exactly TOP_K entries (pad with last entry if needed)
        while len(smis) < CFG.TOP_K:
            smis.append(smis[-1] if smis else "C")

        rows.append({"molecule_id": str(mid), "smiles": ";".join(smis)})

    # ── Write submission.csv ─────────────────────────────────────────────────
    sub = pd.DataFrame(rows)

    # Align row order to sample_submission.csv
    sample_path = CFG.RAW_DIR / "sample_submission.csv"
    if sample_path.is_file():
        sample = pd.read_csv(sample_path)
        sub = sample[["molecule_id"]].merge(sub, on="molecule_id", how="left")
        log.info("Row order aligned to sample_submission.csv.")

    out_path = CFG.ROOT / "predictions" / "submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    log.info(f"Wrote {out_path}  ({len(sub)} rows × {CFG.TOP_K} candidates).")


if __name__ == "__main__":
    main()
