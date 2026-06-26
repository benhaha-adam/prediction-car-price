# AutoPredict — Deployment Guide

## Local run (test before deploying)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy on Streamlit Community Cloud (free)

### Step 1 — Prepare your repo

Your GitHub repo must contain these files:

```
your-repo/
├── app.py                        ← this file
├── requirements.txt              ← this file
└── models/
    ├── training_metadata.json    ← from train.py output
    ├── random_forest.joblib      ← your trained models
    ├── gradient_boosting.joblib
    └── ...
```

> ⚠️ The `models/` folder must be committed to the repo.
> If `.joblib` files are large (>100 MB), use Git LFS:
> `git lfs track "*.joblib"`

### Step 2 — Push to GitHub

```bash
git init
git add app.py requirements.txt models/
git commit -m "AutoPredict app"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 3 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo, branch (`main`), and main file (`app.py`)
5. Click **Deploy** — your app will be live in ~2 minutes

Your public URL will be:
`https://YOUR_USERNAME-YOUR_REPO-app-XXXXX.streamlit.app`

---

## Keeping models out of git (alternative)

If your `.joblib` files are too large for GitHub, upload them to
an S3 bucket or Hugging Face Hub and add a `startup.py` that
downloads them on first boot. Ask Claude for help setting that up.