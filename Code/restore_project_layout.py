"""Restore original code paths so relative imports and file references are preserved."""
from pathlib import Path
import argparse, csv, shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='Reproduction', help='New working directory')
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    target = Path(args.target).resolve()
    with (package / 'file_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        entries = list(csv.DictReader(f))
    # Check all destinations before writing; an existing project is never overwritten.
    for row in entries:
        dest = target / row['project_path']
        if dest.exists():
            raise FileExistsError(f'Choose an empty working directory: {dest}')
    for row in entries:
        dest = target / row['project_path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / row['package_path'], dest)
    print(f'Restored {len(entries)} files to {target}')

if __name__ == '__main__':
    main()
