from pathlib import Path
import redivis
from tqdm import tqdm

# Output directory
out_dir = Path("/work/yilmaz/")
out_dir.mkdir(parents=True, exist_ok=True)

table = redivis.table(
    "sdss_data_repository.stanford_2d_3d_semantics_dataset_2d_3d_s:f304:v1_0.no_xyz:ct1f"
)

# List raw files from the file-index table
files = table.list_files(max_results=None)

print(f"Found {len(files)} files")

for f in tqdm(files):
    # Redivis stores original path/name in f.path / f.name
    dst = out_dir / str(f.path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        continue

    f.download(str(dst))
