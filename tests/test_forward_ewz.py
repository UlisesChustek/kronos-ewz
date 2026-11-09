import sys
import unittest
import tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import forward_ewz as forward

class ForwardTest(unittest.TestCase):
    def frame(self,n):
        return pd.DataFrame(dict(timestamps=pd.date_range("2030-01-01 10:00",periods=n,freq="h"),close=[101+i for i in range(n)]))
    def record(self):
        return dict(anchor="2030-01-01 09:00:00",issued_utc="2030-01-01T09:59:59+00:00",close0=100,predicted_return=.01)
    def test_requires_seven_complete_targets(self):
        self.assertIsNone(forward.mature(self.record(),self.frame(6)))
        result=forward.mature(self.record(),self.frame(7))
        self.assertAlmostEqual(result["actual_return"],.07)
        self.assertTrue(result["fully_prospective"])
    def test_later_bars_cannot_change_matured_outcome(self):
        first=forward.mature(self.record(),self.frame(7))
        later=self.frame(12)
        later.loc[7:,"close"]=900
        self.assertEqual(first,forward.mature(self.record(),later))
    def test_partially_elapsed_target_is_classified(self):
        record=self.record()
        record["issued_utc"]="2030-01-01T10:30:00+00:00"
        self.assertFalse(forward.mature(record,self.frame(7))["fully_prospective"])
    def test_atomic_write_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"ledger.json"
            data=dict(forecasts=[],outcomes=[])
            forward.atomic_json(path,data)
            first=path.read_bytes()
            forward.atomic_json(path,data)
            self.assertEqual(first,path.read_bytes())

if __name__=="__main__":
    unittest.main()
