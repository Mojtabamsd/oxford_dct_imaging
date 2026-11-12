"""Streamlit app for the plankton challenge leaderboard (runs on Streamlit Community Cloud).

Run locally for testing:
    CHALLENGE_LOCAL_DIR=path/to/private/files streamlit run leaderboard/streamlit_app.py
"""

from __future__ import annotations

import os

import streamlit as st

# On Streamlit Cloud, DATA_REPO and HF_TOKEN come from the app's Secrets settings.
try:
    for _key in ("DATA_REPO", "HF_TOKEN"):
        if _key in st.secrets and _key not in os.environ:
            os.environ[_key] = str(st.secrets[_key])
except Exception:  # no secrets file (local test with CHALLENGE_LOCAL_DIR)  # noqa: BLE001
    pass

from backend import COLUMNS, Challenge, SubmissionError  # noqa: E402

st.set_page_config(page_title="Plankton challenge leaderboard", page_icon="🔬", layout="wide")


@st.cache_resource
def get_challenge() -> Challenge:
    """One shared instance per server: loads the answer key and submission log once."""
    return Challenge.from_env()


challenge = get_challenge()

st.title("Teaching machines to see plankton")
st.caption("Upload your predictions on the NOC UVP6 test set and see how your model does on images "
           "from a new deployment. The labels are hidden; only this page can score them.")

board_tab, submit_tab, rules_tab = st.tabs(["🏆 Leaderboard", "📤 Submit", "ℹ️ How it works"])

with board_tab:
    board = challenge.leaderboard()
    if board.empty:
        st.info("No submissions yet. Be the first!")
    else:
        st.dataframe(board, hide_index=True, width="stretch")
    st.caption("Each team's best submission, ranked by macro F1.")

with submit_tab:
    with st.form("submit", clear_on_submit=False):
        col1, col2 = st.columns(2)
        team = col1.selectbox("Team", challenge.team_names, index=None, placeholder="Choose your team")
        code = col2.text_input("Team code", type="password")
        upload = st.file_uploader("predictions.csv", type=["csv"])
        sent = st.form_submit_button("Submit", type="primary")

    if sent:
        try:
            with st.spinner("Scoring..."):
                result = challenge.submit(team, code, upload.getvalue() if upload else b"")
        except SubmissionError as err:
            st.error(f"Not scored: {err}")
        else:
            m = result["metrics"]
            st.success(f"Scored: macro F1 **{m['unweighted']['macro_f1']:.4f}**. "
                       f"Submissions left today: {result['remaining_today']}.")
            shown = [(heading, m.get(kind, {}).get(metric)) for col, (heading, (kind, metric)) in COLUMNS.items()]
            shown = [(h, v) for h, v in shown if v is not None]
            for col, (heading, value) in zip(st.columns(len(shown)), shown):
                col.metric(heading, f"{value:.3f}")
            if challenge.settings["show_per_class"]:
                st.subheader("Per-class results")
                st.dataframe(result["table"][["precision", "recall", "f1"]].round(3), width="stretch")
            st.caption("The leaderboard tab updates the next time you open it or refresh the page.")

with rules_tab:
    st.markdown(f"""
### Submission format
A CSV with one row per test image (all {len(challenge.truth)} of them) and two columns:

```
object_id,label
20240609-200907-1_65,detritus
20240609-200907-1_75,Copepoda
```

* `label` is a UVP6Net class name at **level 2** (32 classes) or **level 1** (54 classes, mapped to
  level 2 automatically). The list is in `configs/uvp6net_classes.csv` in the challenge repository.
* The test set contains **one class that is not in the training data**. Label objects you think
  belong to it as `unknown`.

### Scoring
* **Macro F1** (ranking metric): F1 averaged over classes, each class counting equally, with
  `unknown` counted as one class. Accuracy isn't used, because predicting "detritus" for
  everything already scores high.
* **Closed-set macro F1**: the same, but only on images of classes seen in training.
* **Unknown recall**: share of the novel-class images you labelled `unknown`.
* **False-unknown rate**: share of known-class images you wrongly labelled `unknown`.
* **Weighted macro F1**: the test images weren't sampled at random (rare classes were
  over-sampled). Weighting undoes this, estimating performance on the full deployment.

### Rules
* Up to **{challenge.settings['max_per_day']} scored submissions per team per day** (UTC). Failed
  uploads don't count.
* The leaderboard shows each team's **best** submission.
* Tune on the UVP6Net validation split rather than this leaderboard:
  `python scripts/evaluate.py preds.csv --truth uvp6net-valid` scores it locally, as often as you like.
""")
    if challenge.deadline:
        st.markdown(f"* Submissions close at **{challenge.deadline:%A %d %B %Y, %H:%M} UTC**.")
