# Leaderboard setup (organisers only)

The leaderboard is a small web page, hosted free on
[Streamlit Community Cloud](https://streamlit.io/cloud). Teams upload `predictions.csv` there and
are scored against the hidden NOC labels. Students don't need accounts: each team gets a short
team code instead.

```
students ──upload CSV──▶  Streamlit app (public page)  ──reads/writes──▶  private HF dataset
                         scores, shows ranking                         answer key, team codes,
                         code: leaderboard/ in this repo               settings, all submissions
```

The app's code is public, in this repository, and contains no secrets. The answer key lives only
in the private Hugging Face dataset.

## One-time setup

### 1. Private data on Hugging Face
```bash
pip install huggingface_hub          # Python 3.10+
hf auth login
python scripts/maintainer/setup_leaderboard.py --answer-key data/release/answer_key.csv --teams 6
```
This creates `<your-hf-user>/plankton-challenge-private`, and prints the team codes and the
secrets the app needs.

### 2. A token for the app
On Hugging Face, go to *Settings → Access Tokens → Create new token → Fine-grained*.
* **Repository permissions:** add the dataset `plankton-challenge-private`, and tick *read* and *write*.
* No other permissions are needed.

Copy the token (`hf_...`).

### 3. The app on Streamlit Community Cloud
1. Push this repository to GitHub.
2. Sign in at <https://share.streamlit.io> with GitHub, then choose *Create app → Deploy a public app from GitHub*:
   * **Repository / branch:** this repo, `main`
   * **Main file path:** `leaderboard/streamlit_app.py`
   * **App URL:** pick something like `plankton-challenge`
   * *Advanced settings:* **Python 3.11**. Under **Secrets**, paste:
     ```toml
     DATA_REPO = "<your-hf-user>/plankton-challenge-private"
     HF_TOKEN = "hf_..."
     ```
3. Click **Deploy**. The first build takes a few minutes.
4. Send each team its code, and share the app URL with everyone.

## Try it locally (optional)
```bash
pip install -r leaderboard/requirements.txt
python scripts/maintainer/setup_leaderboard.py --answer-key data/release/answer_key.csv \
    --local-test leaderboard_test/
CHALLENGE_LOCAL_DIR=leaderboard_test streamlit run leaderboard/streamlit_app.py
```

## Day to day

| Task | How |
|---|---|
| See all submissions | `submissions.csv` in the private dataset; uploaded files are in `submissions/<team>/` |
| Full report for a team's file | `python scripts/evaluate.py <file> --truth data/release/answer_key.csv --out-dir report/` |
| Change the daily limit or set a deadline | re-run `setup_leaderboard.py` with `--max-per-day` / `--deadline`, then *Reboot app* in Streamlit (codes and submissions are kept) |
| Lost team codes | `teams.json` in the private dataset; re-running the setup script also prints them |
| Remove a submission | delete its row from `submissions.csv` in the private dataset, then *Reboot app* |
| App asleep | free apps sleep after a few days without visitors; the first visitor clicks "wake up" (about 30 s) |

## Settings (`settings.json` in the private dataset)

| Key | Default | Meaning |
|---|---|---|
| `max_per_day` | 10 | scored submissions per team per UTC day; failed uploads don't count |
| `deadline` | none | optional UTC time after which submissions are refused |
| `rank_by` | `macro_f1` | metric that picks each team's best and ranks teams |
| `show_per_class` | true | show per-class precision/recall/F1 to the submitting team |

## Notes
* **Only one copy of the app should run.** The submission log is kept in memory and saved to the
  dataset after each submission, so don't deploy the app twice against the same dataset.
* **The daily limit stops probing.** With unlimited submissions, a team could work out the hidden
  labels image by image.
* **Never commit `answer_key.csv` or `validation_merged.csv`.** `.gitignore` already excludes them.
