"""Check requested seeds affect real AMCL library sampling, not just the CLI."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]


class NativeSeeds(unittest.TestCase):
    def test_same_seed_repeats_and_different_seed_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'map').write_text('20 20 .1 -1 -1\n'+' '.join(['-1']*400))
            (p/'frames').write_text('0 0 0 8\n1 0 0 0 3 8 -1 8 0 8 1\n')
            outputs=[]
            for i,seed in enumerate([0,0,1]):
                out=p/str(i)
                subprocess.run(['bash','-c','source "$1"; shift; exec "$@"','bash',
                    str(ROOT/'vendor/setup_amcl.bash'),str(ROOT/'vendor/offline_amcl'),str(p/'map'),
                    str(p/'frames'),str(out),'0','0','0',str(seed)],check=True,capture_output=True)
                outputs.append(out.read_bytes())
            self.assertEqual(outputs[0],outputs[1]);self.assertNotEqual(outputs[0],outputs[2])


if __name__=='__main__':unittest.main()
