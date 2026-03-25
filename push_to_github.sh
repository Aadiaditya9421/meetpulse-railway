#!/bin/bash
# ============================================================
# push_railway.sh — Push capstone-railway to GitHub
# Run this from INSIDE the capstone-railway/ folder
# ============================================================

set -e  # Exit on any error

echo "============================================"
echo "  MeetPulse Railway — GitHub Push Script"
echo "============================================"

# ── Step 1: Collect inputs ────────────────────────────────
read -p "Enter your GitHub username: " GH_USER
read -p "Enter your GitHub repo name (e.g. meetpulse-railway): " GH_REPO
read -p "Enter your GitHub Personal Access Token (PAT): " -s GH_TOKEN
echo ""

REMOTE_URL="https://${GH_USER}:${GH_TOKEN}@github.com/${GH_USER}/${GH_REPO}.git"

# ── Step 2: Git init ──────────────────────────────────────
echo ""
echo "[1/5] Initializing git repo..."
git init
git config user.name "$GH_USER"
git config user.email "${GH_USER}@users.noreply.github.com"

# ── Step 3: .gitignore ───────────────────────────────────
echo "[2/5] Creating .gitignore..."
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pyo
.env
.venv/
venv/
*.egg-info/
dist/
build/
.DS_Store
Thumbs.db
.ipynb_checkpoints/
EOF

echo ""
echo "⚠️  IMPORTANT: Do you want to commit the .pkl model files?"
echo "   (Required for Railway deployment — they must be in the repo)"
read -p "   Commit .pkl files? [y/n]: " COMMIT_PKL

if [ "$COMMIT_PKL" != "y" ]; then
    echo "*.pkl" >> .gitignore
    echo "   Skipping .pkl files."
else
    echo "   .pkl files will be committed."
fi

# ── Step 4: Create GitHub repo via API ───────────────────
echo ""
echo "[3/5] Creating GitHub repository '${GH_REPO}'..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST \
    -H "Authorization: token ${GH_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{
        \"name\": \"${GH_REPO}\",
        \"description\": \"MeetPulse: Video Conferencing Sentiment Analysis — FastAPI + Railway\",
        \"private\": false,
        \"auto_init\": false
    }" \
    "https://api.github.com/user/repos")

if [ "$HTTP_STATUS" = "201" ]; then
    echo "   ✅ GitHub repo created: https://github.com/${GH_USER}/${GH_REPO}"
elif [ "$HTTP_STATUS" = "422" ]; then
    echo "   ℹ️  Repo already exists — continuing..."
else
    echo "   ❌ Failed to create repo (HTTP $HTTP_STATUS). Check your PAT and try again."
    exit 1
fi

# ── Step 5: Commit and push ──────────────────────────────
echo ""
echo "[4/5] Staging and committing files..."
git add .
git commit -m "Initial commit: MeetPulse Railway deployment

- FastAPI backend with /predict, /health, /models/info endpoints
- Bootstrap 5 frontend served as static files
- ML models: LR, SVM, DT, NB, MLP trained on meeting transcripts
- Procfile configured for Railway deployment"

echo ""
echo "[5/5] Pushing to GitHub..."
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"
git push -u origin main

echo ""
echo "============================================"
echo "  ✅ DONE!"
echo "  GitHub: https://github.com/${GH_USER}/${GH_REPO}"
echo ""
echo "  Next: Deploy on Railway"
echo "  1. Go to https://railway.app"
echo "  2. New Project → Deploy from GitHub"
echo "  3. Select: ${GH_USER}/${GH_REPO}"
echo "  4. Railway auto-detects Procfile ✅"
echo "============================================"
