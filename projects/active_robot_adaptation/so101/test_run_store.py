import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_store


class RunStoreTest(unittest.TestCase):
    def test_unique_runs_and_configuration_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run_store,'ROOT',Path(directory)):
            config={'seed':7,'camera_shift_m':[.01,0.]}
            first=run_store.create_run('test',config)
            second=run_store.create_run('test',config)
            self.assertNotEqual(first,second)
            self.assertEqual(json.loads((first/'metadata.json').read_text())['config'],config)
            self.assertTrue((first/'source_snapshot.json.gz').exists())
            self.assertTrue((second/'metadata.json').exists())


if __name__=='__main__': unittest.main()
