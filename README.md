# MeetPulse — FastAPI Deployment (Railway)
**ADITYA SINGH | Roll No: 23052212 | KIIT University**

## Deployed Model: MLPClassifier (F1=0.903)
- Best sklearn model — no TensorFlow dependency
- 2 hidden layers: 256 → 128 neurons, early stopping
- Previous: SVM (F1=0.888) — replaced for +1.5% accuracy improvement
- Same joblib pkl pipeline, zero code changes needed

## Setup
1. Run `notebook.ipynb` on Kaggle (outputs: `model.pkl`, `tfidf.pkl`, `label_encoder.pkl`)
2. Copy the 3 pkl files into this directory
3. Deploy to Railway via Procfile

## API Endpoints
- `GET  /health` — model status + type + F1 score
- `POST /predict` — sentiment prediction (returns `low_confidence` flag if conf < 60%)
- `GET  /models/info` — full model leaderboard
