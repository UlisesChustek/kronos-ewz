"""Fetch candidate Dukascopy BID hourly bars and require historical overlap agreement."""
import argparse
from datetime import datetime, timezone
import json
import lzma
from pathlib import Path
import struct
import urllib.request
import pandas as pd
from forward_ewz import ROOT, BASE, atomic_json, checked_data

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol",default="EWZ.USUSD")
    parser.add_argument("--start",default="2026-10-30")
    parser.add_argument("--end",default=datetime.now(timezone.utc).date().isoformat())
    args=parser.parse_args()
    dates=pd.date_range(args.start,args.end)
    if len(dates)>14 or dates[-1].date()>datetime.now(timezone.utc).date():
        raise ValueError("Collection must cover at most 14 days through today")
    folder=ROOT/"data/forward"
    folder.mkdir(exist_ok=True)
    candidate=folder/"candidate.csv"
    accepted=folder/"continuation.csv"
    now=pd.Timestamp.now(tz="UTC")
    status=dict(at_utc=now.isoformat(), symbol_candidate=args.symbol,
                accepted=False, urls=[], source="Dukascopy BID_candles_hour_1.bi5",
                price_divisor_candidate=1000, verification="requires seven matching historical overlap bars")
    rows=[]
    try:
        for day in dates:
            url=f"https://datafeed.dukascopy.com/datafeed/{args.symbol}/{day.year}/{day.month-1:02d}/{day.day:02d}/BID_candles_hour_1.bi5"
            status["urls"].append(url)
            payload=urllib.request.urlopen(url,timeout=15).read()
            if not payload:
                continue
            decoded=lzma.decompress(payload)
            if len(decoded)%24:
                raise ValueError("Unexpected binary record length")
            for offset in range(0,len(decoded),24):
                seconds,open_,close,low,high,volume=struct.unpack(">5If",decoded[offset:offset+24])
                stamp=day+pd.Timedelta(seconds=seconds)
                if seconds>=86400:
                    raise ValueError("Unexpected candle offset")
                if volume<=0 or stamp+pd.Timedelta(hours=1)>now.tz_localize(None):
                    continue
                close=close/1000
                rows.append(dict(timestamps=stamp,open=open_/1000,high=high/1000,
                                 low=low/1000,close=close,volume=volume,amount=round(close*volume,6)))
        if not rows:
            raise ValueError("No complete nonempty candles returned")
        frame=pd.DataFrame(rows).sort_values("timestamps")
        frame.to_csv(candidate,index=False)
        checked_data(candidate,now)
        frame.to_csv(accepted,index=False)
        status.update(accepted=True,rows=len(frame),file=str(accepted),latest=str(frame.timestamps.iloc[-1]))
    except Exception as error:
        status["error"]=f"{type(error).__name__}: {error}"
    atomic_json(ROOT/"results/forward_collection_status.json",status)
    print(json.dumps(status,indent=2))

if __name__=="__main__":
    main()
