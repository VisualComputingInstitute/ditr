from pathlib import Path
import argparse

import redivis
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Directory where files will be downloaded",
    )
    args = parser.parse_args()

    output_path = args.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    table = redivis.table(
        "sdss_data_repository.stanford_2d_3d_semantics_dataset_2d_3d_s:f304:v1_0.no_xyz:ct1f"
    )

    files = table.list_files(max_results=None)

    print(f"Found {len(files)} files")

    for f in tqdm(files):
        dst = output_path / str(f.path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            continue

        f.download(str(dst))


if __name__ == "__main__":
    main()