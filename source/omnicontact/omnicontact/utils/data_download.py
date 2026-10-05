"""Download motion clips to assets/npz_clips/.

Run from the repository root with: python -m omnicontact.utils.data_download
"""

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError


REPO_ID = "lightcone02/OmniContact-Dataset"


def main() -> None:
    try:
        # Check access before snapshot_download can fall back to existing local files.
        HfApi().auth_check(repo_id=REPO_ID, repo_type="dataset")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=["npz_clips/*"],
            local_dir="assets",
        )
    except GatedRepoError:
        raise SystemExit(
            f"Accept the access conditions at https://huggingface.co/datasets/{REPO_ID}, "
            "then run `hf auth login` with an authorized account and retry."
        ) from None

    print("Motion clips are ready in assets/npz_clips/.")


if __name__ == "__main__":
    main()
