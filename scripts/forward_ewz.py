"""Freeze the existing direct return probe and issue/score timestamped forecasts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import numpy as np
import pandas as pd
import torch
import direct_probe as probe

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "finetuned/direct_frozen_v2"
LEDGER = ROOT / "results/direct_forward_ledger.json"
BASE = ROOT / "data/processed/ewz_h1.csv"

def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)

def models():
    tokenizer = probe.KronosTokenizer.from_pretrained(str(probe.TOKENIZER_DIR), local_files_only=True).to("cuda:1").eval()
    model = probe.Kronos.from_pretrained(str(probe.PREDICTOR_DIR), local_files_only=True).to("cuda:1").eval()
    tokenizer.requires_grad_(False)
    model.requires_grad_(False)
    return tokenizer, model

def representations(frame, anchors, loaded):
    return probe.extract_representations(frame, np.asarray(anchors), *loaded, torch.device("cuda:1"), 64, "frozen")

def freeze():
    if ART.exists():
        raise FileExistsError(f"Frozen artifact already exists: {ART}")
    frame = probe.load_data(BASE)
    anchors = probe.make_anchors(len(frame))
    loaded = models()
    train = representations(frame, anchors["train"], loaded)
    scaler = probe.FeatureScaler().fit(train)
    report = json.loads((ROOT / "results/direct_probe.json").read_text())
    penalty = report["selection"]["repr_ridge"]["chosen"]["lambda"]
    targets = probe.target_returns(frame, anchors["train"])
    beta = probe.fit_ridge(scaler.transform(train), targets, penalty)
    test = representations(frame, anchors["test"], loaded)
    predictions = probe.predict_ridge(scaler.transform(test), beta)
    mae = float(np.abs(predictions - probe.target_returns(frame, anchors["test"])).mean())
    expected = report["test"]["models"]["repr_ridge"]["terminal_mae"]
    if not np.isclose(mae, expected, atol=1e-9, rtol=1e-6):
        raise ValueError(f"Replay mismatch {mae} vs {expected}")
    ART.mkdir(parents=True)
    np.savez(ART / "head.npz", mean=scaler.mean_, std=scaler.std_, beta=beta)
    paths = [BASE, ROOT/"scripts/direct_probe.py", Path(__file__),
             ROOT/"scripts/walk_forward_baselines.py",
             ROOT/"third_party/Kronos/model/kronos.py", ROOT/"third_party/Kronos/model/module.py"]
    for directory in (probe.PREDICTOR_DIR, probe.TOKENIZER_DIR):
        paths.extend(directory.glob("*.json"))
        paths.extend(directory.glob("*.safetensors"))
    manifest = dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
                    cutoff=str(frame.timestamps.iloc[-1]), rows=len(frame), lookback=128,
                    horizon=7, penalty=penalty, train_prior=float((targets>0).mean()),
                    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths},
                    head_sha256=digest(ART/"head.npz"), replay_mae=mae,
                    normalization="past128 mean/std+1e-5, clip5; train scaler clip8",
                    fit="original training labels only; fixed prior validation penalty")
    atomic_json(ART/"manifest.json", manifest)
    print(json.dumps(manifest, indent=2))

def checked_data(continuation, now):
    base = probe.load_data(BASE)
    if continuation:
        extra = probe.load_data(continuation)
        if extra.timestamps.duplicated().any():
            raise ValueError("Duplicate continuation timestamps")
        overlap = extra.merge(base, on="timestamps", suffixes=("_new","_base"))
        if len(overlap) < 7:
            raise ValueError("Continuation requires at least seven matching overlap bars")
        for col in probe.FEATURES:
            np.testing.assert_allclose(overlap[col+"_new"], overlap[col+"_base"], rtol=1e-6, atol=1e-6)
        base = pd.concat([base, extra[extra.timestamps>base.timestamps.iloc[-1]]],ignore_index=True)
    numeric = base[probe.FEATURES].to_numpy()
    if not np.isfinite(numeric).all() or (base.close<=0).any():
        raise ValueError("Invalid numeric bars")
    return base[base.timestamps + pd.Timedelta(hours=1) <= now.tz_localize(None)].reset_index(drop=True)

def mature(record, frame):
    # Keep the original seven-bar target anchored to the last input bar.
    # Classify the actual target timing against issuance when it matures.
    issued = pd.Timestamp(record["issued_utc"]).tz_convert(None)
    after = frame[(frame.timestamps > pd.Timestamp(record["anchor"]))]
    if len(after)<7:
        return None
    actual = float(after.close.iloc[6]/record["close0"]-1)
    return dict(actual_return=actual, terminal_timestamp=str(after.timestamps.iloc[6]),
                fully_prospective=bool(after.timestamps.iloc[0] >= issued),
                first_target_timestamp=str(after.timestamps.iloc[0]),
                absolute_error=abs(record["predicted_return"]-actual), zero_return_error=abs(actual))

def update(continuation):
    manifest = json.loads((ART/"manifest.json").read_text())
    if digest(ART/"head.npz") != manifest["head_sha256"]:
        raise ValueError("Frozen head changed")
    for rel, expected in manifest["hashes"].items():
        if digest(ROOT/rel) != expected:
            raise ValueError(f"Frozen input/code changed: {rel}")
    now = pd.Timestamp.now(tz="UTC")
    frame = checked_data(continuation, now)
    fingerprint = digest(ART/"manifest.json")
    LEDGER.parent.mkdir(exist_ok=True)
    with open(str(LEDGER)+".lock","a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(model=fingerprint, forecasts=[], outcomes=[])
        if ledger["model"] != fingerprint:
            raise ValueError("Ledger model fingerprint differs")
        scored = {item["forecast_id"] for item in ledger["outcomes"]}
        for record in ledger["forecasts"]:
            history = frame[frame.timestamps <= pd.Timestamp(record["anchor"])].tail(128)
            context_hash = hashlib.sha256(history.to_csv(index=False).encode()).hexdigest()
            if context_hash != record["context_sha256"]:
                raise ValueError("Previously forecast context was revised")
            if record["id"] in scored:
                old = next(row for row in ledger["outcomes"] if row["forecast_id"]==record["id"])
                current = mature(record, frame)
                if current is None or current["actual_return"] != old["actual_return"]:
                    raise ValueError("Previously scored target was revised")
            if record["id"] not in scored:
                outcome = mature(record, frame)
                if outcome:
                    ledger["outcomes"].append(dict(forecast_id=record["id"], scored_utc=now.isoformat(), **outcome))
        anchor = str(frame.timestamps.iloc[-1])
        key = fingerprint+":"+anchor
        if not any(row["id"]==key for row in ledger["forecasts"]):
            if frame.timestamps.iloc[-1] <= pd.Timestamp(manifest["cutoff"]):
                print("Waiting for validated post-cutoff bars; no stale forecast issued.")
            else:
                loaded = models()
                feature = representations(frame, [len(frame)], loaded)
                with np.load(ART/"head.npz", allow_pickle=False) as head:
                    transformed = np.clip((feature-head["mean"])/head["std"],-8,8)
                    predicted = float(probe.predict_ridge(transformed, head["beta"])[0])
                ledger["forecasts"].append(dict(id=key, issued_utc=now.isoformat(), anchor=anchor,
                    close0=float(frame.close.iloc[-1]), predicted_return=predicted,
                    context_sha256=hashlib.sha256(frame.tail(128).to_csv(index=False).encode()).hexdigest(),
                    target_rule="next seven complete supplied bars after anchor; classify target timing on maturity"))
        atomic_json(LEDGER, ledger)
        print(json.dumps(ledger,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["freeze","update"])
    parser.add_argument("--continuation",type=Path)
    args=parser.parse_args()
    if args.action=="freeze":
        freeze()
    else:
        update(args.continuation)
