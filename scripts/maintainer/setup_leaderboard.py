#!/usr/bin/env python
"""Set up the leaderboard's private data on Hugging Face (maintainer only).

Creates (or updates) a PRIVATE Hugging Face dataset ``<user>/<name>-private``
holding the answer key, team codes and settings. The leaderboard app
(leaderboard/streamlit_app.py, hosted on Streamlit Community Cloud) reads it and
stores every submission there. See leaderboard/SETUP.md.

Safe to re-run: team codes and submissions are kept unless you pass
``--new-team-codes``. Re-run to change settings, then reboot the app.

Setup (once):
    pip install huggingface_hub
    hf auth login

Examples:
    python scripts/maintainer/setup_leaderboard.py --answer-key data/release/answer_key.csv --teams 6

    # Local test without Hugging Face:
    python scripts/maintainer/setup_leaderboard.py --answer-key data/release/answer_key.csv \
        --local-test leaderboard_test/
    CHALLENGE_LOCAL_DIR=leaderboard_test streamlit run leaderboard/streamlit_app.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import secrets
from datetime import datetime
from pathlib import Path


def make_code() -> str:
    # 3 groups of 4 characters without look-alikes (0/o, 1/l/i): easy to type, ~59 bits
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    return "-".join("".join(secrets.choice(alphabet) for _ in range(4)) for _ in range(3))


def merge_codes(names: list[str], old: dict[str, str] | None) -> dict[str, str]:
    """Keep codes of teams that already have one; create codes for new teams."""
    old = old or {}
    return {n: old.get(n) or make_code() for n in names}


def print_codes(teams: dict[str, str]) -> None:
    print("\nTeam codes (send each team only its own; they are also in teams.json):")
    for name, code in teams.items():
        print(f"  {name:<12} {code}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answer-key", type=Path, required=True, help="answer_key.csv from build_noc_release.py")
    parser.add_argument("--name", default="plankton-challenge",
                        help="dataset becomes <user>/<name>-private (default: plankton-challenge)")
    parser.add_argument("--teams", type=int, default=6, help="number of teams (default: 6)")
    parser.add_argument("--team-names", nargs="+", help="custom team names instead of 'Team 1' ... 'Team N'")
    parser.add_argument("--max-per-day", type=int, default=10, help="scored submissions per team per day")
    parser.add_argument("--deadline", help="optional: close submissions at this UTC time, e.g. 2027-01-13T23:59")
    parser.add_argument("--new-team-codes", action="store_true", help="replace all existing team codes")
    parser.add_argument("--local-test", type=Path, metavar="DIR",
                        help="write the private files to DIR for a local test instead of Hugging Face")
    args = parser.parse_args()

    if args.deadline:
        datetime.fromisoformat(args.deadline)  # fail early on a typo
    names = args.team_names or [f"Team {i}" for i in range(1, args.teams + 1)]
    settings = json.dumps({"max_per_day": args.max_per_day, "deadline": args.deadline}, indent=2)
    key = args.answer_key.read_bytes()

    if args.local_test:
        out = args.local_test
        out.mkdir(parents=True, exist_ok=True)
        teams_path = out / "teams.json"
        old = None if args.new_team_codes or not teams_path.exists() else json.loads(teams_path.read_text())
        teams = merge_codes(names, old)
        (out / "answer_key.csv").write_bytes(key)
        (out / "settings.json").write_text(settings)
        teams_path.write_text(json.dumps(teams, indent=2))
        print(f"Local test files in {out.resolve()}")
        print(f"Run:  CHALLENGE_LOCAL_DIR={out} streamlit run leaderboard/streamlit_app.py")
        print_codes(teams)
        return

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    api = HfApi()
    data_id = f"{api.whoami()['name']}/{args.name}-private"
    api.create_repo(data_id, repo_type="dataset", private=True, exist_ok=True)

    old = None
    if not args.new_team_codes:
        with contextlib.suppress(EntryNotFoundError):
            old = json.loads(Path(hf_hub_download(data_id, "teams.json", repo_type="dataset",
                                                  force_download=True)).read_text())
    teams = merge_codes(names, old)

    files = {"answer_key.csv": key, "settings.json": settings.encode(),
             "teams.json": json.dumps(teams, indent=2).encode()}
    api.create_commit(data_id, [CommitOperationAdd(p, d) for p, d in files.items()],
                      commit_message="Update leaderboard data", repo_type="dataset")

    print(f"Private dataset: https://huggingface.co/datasets/{data_id}")
    print("\nStreamlit app secrets (paste into the app's Settings -> Secrets):\n")
    print(f'DATA_REPO = "{data_id}"')
    print('HF_TOKEN = "hf_..."   # a fine-grained token with read+write access to the dataset above')
    print_codes(teams)


if __name__ == "__main__":
    main()
