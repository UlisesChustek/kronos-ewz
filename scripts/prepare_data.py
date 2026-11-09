from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
TFS = ["d1", "h1"]


def load_tf(tf: str) -> pd.DataFrame:
    files = sorted((ROOT / "data" / tf).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under data/{tf}")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"time": "timestamps"})
    df = df[["timestamps", "open", "high", "low", "close", "volume"]].copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)
    df = df.sort_values("timestamps").drop_duplicates("timestamps").reset_index(drop=True)
    df["amount"] = (df["close"] * df["volume"]).round(6)
    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    return df, files


def validate(df: pd.DataFrame, tf: str) -> None:
    assert df["timestamps"].is_monotonic_increasing, f"{tf}: timestamps not sorted"
    assert not df["timestamps"].duplicated().any(), f"{tf}: duplicate timestamps"
    assert not df.isna().any().any(), f"{tf}: null values"
    assert (df[["open", "high", "low", "close"]] > 0).all().all(), f"{tf}: non-positive prices"
    invalid = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    assert not invalid.any(), f"{tf}: invalid OHLC rows"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tf in TFS:
        df, files = load_tf(tf)
        validate(df, tf)
        csv_path = OUT_DIR / f"ewz_{tf}.csv"
        out = df.copy()
        out["timestamps"] = out["timestamps"].dt.strftime("%Y-%m-%d %H:%M:%S")
        out.to_csv(csv_path, index=False)
        meta = {
            "instrument": "EWZ.US/USD",
            "source": "Dukascopy (ETF CFD, BID)",
            "timeframe": tf,
            "source_files": [str(f.relative_to(ROOT)) for f in files],
            "rows": int(len(out)),
            "first": str(out["timestamps"].iloc[0]),
            "last": str(out["timestamps"].iloc[-1]),
            "columns": list(out.columns),
            "amount_proxy": "close * volume (Dukascopy CFD feed volume)",
            "volume_note": "CFD feed volume, not consolidated exchange volume",
            "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        }
        meta_path = OUT_DIR / f"ewz_{tf}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"{tf}: {len(out)} rows -> {csv_path}")
        print(f"    {meta['first']} -> {meta['last']}")


if __name__ == "__main__":
    main()
