"""Download only declared development inputs; never fetch reserved final data."""
import subprocess
import zipfile
from pathlib import Path

BASE = 'https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/data/'
FILES = {
    'map_office.zip': 'map_office.zip',
    'dynamics_0.bag': 'bagfiles_no_cams/dynamics_0.bag',
    'lt_changes_0.bag': 'bagfiles_no_cams/lt_changes_0.bag',
    'dynamics_0_gt.txt': 'gt_poses_public/dynamics_0/gt_poses.txt',
    'lt_changes_0_gt.txt': 'gt_poses_public/lt_changes_0/gt_poses.txt',
}

if __name__ == '__main__':
    root = Path(__file__).resolve().parent/'data'
    root.mkdir(exist_ok=True)
    for name, url in FILES.items():
        path = root/name
        if not path.exists():
            temporary = path.with_suffix(path.suffix+'.part')
            subprocess.run(['curl','--fail','--location','--max-time','180',BASE+url,
                            '-o',str(temporary)],check=True)
            temporary.replace(path)
    with zipfile.ZipFile(root/'map_office.zip') as archive:
        # Exact provider manifest; no untrusted arbitrary extraction paths.
        assert set(archive.namelist()) == {'map_office.yaml','map_office.png'}
        archive.extractall(root/'map')
