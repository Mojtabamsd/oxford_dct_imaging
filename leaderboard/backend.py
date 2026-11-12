"""Leaderboard backend: team codes, submission limits, scoring and storage.

All private state lives in a *private* Hugging Face dataset repo (``DATA_REPO``):

    answer_key.csv         object_id,expert_label,label,weight   (hidden labels)
    teams.json             {"Team 1": "<code>", ...}
    settings.json          {"max_per_day": 10, "deadline": null, ...}
    submissions.csv        one row per scored submission (the leaderboard log)
    submissions/<team>/... the uploaded prediction files, for the organisers

For local testing, set ``CHALLENGE_LOCAL_DIR`` to a folder with the same files
instead of ``DATA_REPO``.
"""

from __future__ import annotations

import hmac
import io
import json
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
# On the Space, evaluate.py and the class list sit next to this file; in the repo
# they live in scripts/ and configs/.
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
from evaluate import SubmissionError, clean_predictions, evaluate, load_classes, read_predictions  # noqa: E402

CLASSES_FILE = next(p for p in (HERE / "uvp6net_classes.csv", HERE.parent / "configs" / "uvp6net_classes.csv")
                    if p.exists())
MAX_UPLOAD_BYTES = 5_000_000

DEFAULT_SETTINGS = {
    "max_per_day": 10,          # scored submissions per team per UTC day
    "deadline": None,           # ISO time in UTC, e.g. "2027-01-13T23:59:00"; None = always open
    "rank_by": "macro_f1",      # column used to pick each team's best and to rank teams
    "show_per_class": True,     # show per-class precision/recall/F1 to the submitting team
}

# Leaderboard column -> (heading, source in the metrics dict)
COLUMNS = {
    "macro_f1": ("Macro F1", ("unweighted", "macro_f1")),
    "closed_macro_f1": ("Closed-set macro F1", ("unweighted", "closed_macro_f1")),
    "unknown_recall": ("Unknown recall", ("unweighted", "unknown_recall")),
    "false_unknown_rate": ("False-unknown rate", ("unweighted", "false_unknown_rate")),
    "balanced_accuracy": ("Balanced accuracy", ("unweighted", "balanced_accuracy")),
    "weighted_macro_f1": ("Weighted macro F1", ("weighted", "macro_f1")),
}
LOG_COLUMNS = ["time", "team", *COLUMNS, "file"]


# ---- storage ----------------------------------------------------------------------

class ConfigError(RuntimeError):
    """The app is misconfigured (secrets, token permissions); safe to show to the organiser."""


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def read(self, name: str) -> bytes | None:
        path = self.root / name
        return path.read_bytes() if path.exists() else None

    def write(self, files: dict[str, bytes], message: str) -> None:
        for name, data in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


class HubStore:
    """Files in a private Hugging Face dataset repo."""

    def __init__(self, repo_id: str, token: str | None):
        from huggingface_hub import HfApi
        self.api, self.repo_id = HfApi(token=token), repo_id
        self.check_access(token)

    def check_access(self, token: str | None) -> None:
        """Fail early with a message that says what is misconfigured (never the token itself)."""
        if not token:
            raise ConfigError("HF_TOKEN is not set. Add it to the app's Secrets.")
        try:
            user = self.api.whoami()["name"]
        except Exception as err:  # noqa: BLE001
            raise ConfigError(f"HF_TOKEN was rejected by Hugging Face ({type(err).__name__}). "
                              "Check it was copied in full, or create a new one.") from err
        try:
            self.api.repo_info(self.repo_id, repo_type="dataset")
        except Exception as err:  # noqa: BLE001
            raise ConfigError(
                f"The token (user '{user}') cannot see the dataset '{self.repo_id}' ({type(err).__name__}). "
                "Check DATA_REPO is spelled exactly right, and that the token has read and write "
                "permission on that repository (fine-grained token -> Repositories permissions).") from err

    def read(self, name: str) -> bytes | None:
        from huggingface_hub.utils import EntryNotFoundError
        try:
            path = self.api.hf_hub_download(self.repo_id, name, repo_type="dataset", force_download=True)
        except EntryNotFoundError:
            return None
        return Path(path).read_bytes()

    def write(self, files: dict[str, bytes], message: str) -> None:
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in files.items()]
        self.api.create_commit(self.repo_id, ops, commit_message=message, repo_type="dataset")


def store_from_env():
    if os.environ.get("CHALLENGE_LOCAL_DIR"):
        return LocalStore(Path(os.environ["CHALLENGE_LOCAL_DIR"]))
    repo = os.environ.get("DATA_REPO", "").strip()
    if not repo:
        raise ConfigError("DATA_REPO is not set. Add it to the app's Secrets "
                          '(e.g. DATA_REPO = "<user>/plankton-challenge-private").')
    return HubStore(repo, os.environ.get("HF_TOKEN", "").strip() or None)


# ---- challenge ----------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "team"


class Challenge:
    def __init__(self, store):
        self.store = store
        self.lock = threading.Lock()
        self.classes = load_classes(CLASSES_FILE)

        key = store.read("answer_key.csv")
        teams = store.read("teams.json")
        if key is None or teams is None:
            raise RuntimeError("answer_key.csv and teams.json must exist in the data store.")
        truth = pd.read_csv(io.BytesIO(key), dtype={"object_id": str})
        self.truth = truth[[c for c in ("object_id", "label", "weight") if c in truth.columns]]
        self.teams: dict[str, str] = json.loads(teams)
        settings = store.read("settings.json")
        self.settings = {**DEFAULT_SETTINGS, **(json.loads(settings) if settings else {})}

        log = store.read("submissions.csv")
        self.log = pd.read_csv(io.BytesIO(log)) if log else pd.DataFrame(columns=LOG_COLUMNS)

    @classmethod
    def from_env(cls) -> "Challenge":
        return cls(store_from_env())

    @property
    def team_names(self) -> list[str]:
        return list(self.teams)

    @property
    def deadline(self) -> datetime | None:
        d = self.settings.get("deadline")
        return datetime.fromisoformat(d).replace(tzinfo=timezone.utc) if d else None

    def used_today(self, team: str) -> int:
        if self.log.empty:
            return 0
        today = _now().date().isoformat()
        return int(((self.log["team"] == team) & (self.log["time"].str[:10] == today)).sum())

    def submit(self, team: str, code: str, data: bytes) -> dict:
        """Score one upload. Raises SubmissionError with a user-facing message on any problem."""
        if team not in self.teams:
            raise SubmissionError("Please choose your team.")
        if not code or not hmac.compare_digest(code.strip(), self.teams[team]):
            raise SubmissionError("Wrong team code.")
        if self.deadline and _now() > self.deadline:
            raise SubmissionError(f"Submissions closed at {self.deadline:%Y-%m-%d %H:%M} UTC.")
        if not data:
            raise SubmissionError("Please upload your predictions CSV.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise SubmissionError("File too large; expected a CSV with one row per test image.")

        with self.lock:
            limit = int(self.settings["max_per_day"])
            if self.used_today(team) >= limit:
                raise SubmissionError(f"{team} has used all {limit} submissions for today (UTC). "
                                      "Failed uploads do not count; try again tomorrow.")

            pred = clean_predictions(read_predictions(io.BytesIO(data)), 2, self.classes)
            result = evaluate(self.truth, pred, 2, self.classes)

            now = _now()
            fname = f"submissions/{_slug(team)}/{now:%Y%m%dT%H%M%S_%fZ}.csv"
            row = {"time": now.isoformat(timespec="seconds"), "team": team, "file": fname}
            for col, (_, (kind, metric)) in COLUMNS.items():
                row[col] = result["metrics"].get(kind, {}).get(metric)
            log = pd.concat([self.log, pd.DataFrame([row])], ignore_index=True)[LOG_COLUMNS]

            self.store.write({fname: data, "submissions.csv": log.to_csv(index=False).encode()},
                             message=f"Submission from {team}")
            self.log = log

        result["row"] = row
        result["remaining_today"] = limit - self.used_today(team)
        return result

    def leaderboard(self) -> pd.DataFrame:
        """Each team's best submission, ranked."""
        rank_by = self.settings["rank_by"]
        headings = {col: heading for col, (heading, _) in COLUMNS.items()}
        if self.log.empty:
            return pd.DataFrame(columns=["Rank", "Team", *headings.values(), "Submissions", "Best submitted (UTC)"])

        log = self.log.copy()
        log[list(COLUMNS)] = log[list(COLUMNS)].apply(pd.to_numeric)
        counts = log.groupby("team").size()
        best = log.sort_values([rank_by, "time"], ascending=[False, True]).drop_duplicates("team")
        best = best.sort_values(rank_by, ascending=False).reset_index(drop=True)
        best.insert(0, "Rank", range(1, len(best) + 1))
        best["Submissions"] = best["team"].map(counts)
        best["Best submitted (UTC)"] = best["time"].str.replace("T", " ").str[:16]
        best = best.drop(columns=["time", "file"]).rename(columns={"team": "Team", **headings})
        cols = [c for c in best.columns if best[c].notna().any()]  # drop metrics this test set lacks
        return best[cols].round(4)
