"""
MeetPulse — Video Conferencing Sentiment Analysis API
FastAPI Backend | ADITYA SINGH | Roll No: 23052212 | KIIT University

Primary Model  : MLPClassifier (F1=0.8805, Acc=0.8804)
Fallback Model : SVC Linear   (F1=0.7596, Acc=0.7594)

v3.0 Improvements:
  ✅ MLP primary, SVM explicit fallback with graceful switching
  ✅ /predict/explain — top TF-IDF feature contributions
  ✅ /predict/compare — run both MLP & SVM, return side-by-side
  ✅ /metrics endpoint — live request stats, sentiment histogram
  ✅ Confidence levels: high (>=70%), moderate (55-70%), low (<55%)
  ✅ Startup pre-warm (first request not cold)
  ✅ Request IDs in all responses for traceability
"""

import logging
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import nltk
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("meetpulse")

# ── Auto-download PKL files from HuggingFace if missing ───────────────────
HF_REPO   = "aadiaditya9421/meetpulse"
HF_BASE   = f"https://huggingface.co/{HF_REPO}/resolve/main"

# Map local filename → HF filename (svm_tfidf/svm_label_encoder reuse the same files)
HF_FILES = {
    "model.pkl":           "model.pkl",
    "tfidf.pkl":           "tfidf.pkl",
    "label_encoder.pkl":   "label_encoder.pkl",
    "svm_model.pkl":       "svm_model.pkl",
    "svm_tfidf.pkl":       "tfidf.pkl",          # same vectorizer
    "svm_label_encoder.pkl": "label_encoder.pkl", # same encoder
}

def _download_if_missing(model_dir: Path):
    import urllib.request
    for local_name, hf_name in HF_FILES.items():
        dest = model_dir / local_name
        if dest.exists():
            logger.info("✅ Found: %s", local_name)
            continue
        url = f"{HF_BASE}/{hf_name}"
        logger.info("⬇️  Downloading %s from HuggingFace …", local_name)
        try:
            urllib.request.urlretrieve(url, dest)
            logger.info("✅ Downloaded: %s (%.1f MB)", local_name, dest.stat().st_size / 1e6)
        except Exception as e:
            logger.warning("⚠️  Could not download %s: %s", local_name, e)

app = FastAPI(
    title="MeetPulse API v3",
    description="Sentiment analysis: MLP primary (F1=0.8805), SVM fallback (F1=0.7596). No TF dependency.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MODEL_F1_MAP = {
    "MLPClassifier": 0.8805,
    "SVC": 0.7596,
    "LogisticRegression": 0.7465,
    "DecisionTreeClassifier": 0.4917,
    "MultinomialNB": 0.7066,
}

MODEL_DIR = Path(__file__).parent

# Download any missing pkl files from HuggingFace at startup
_download_if_missing(MODEL_DIR)

def _load_pkl(path: Path):
    try:
        obj = joblib.load(path)
        logger.info("Loaded: %s", path.name)
        return obj
    except FileNotFoundError:
        logger.warning("Not found: %s", path.name)
        return None
    except Exception as e:
        logger.error("Load error %s: %s", path.name, e)
        return None

def _load_set(prefix=""):
    m  = _load_pkl(MODEL_DIR / f"{prefix}model.pkl")
    t  = _load_pkl(MODEL_DIR / f"{prefix}tfidf.pkl")
    le = _load_pkl(MODEL_DIR / f"{prefix}label_encoder.pkl")
    if m and t and le:
        return m, t, le, type(m).__name__
    return None, None, None, None

mlp_model, mlp_tfidf, mlp_le, mlp_name = _load_set("")
svm_model, svm_tfidf, svm_le, svm_name = _load_set("svm_")

if svm_model is None:
    logger.info("svm_model.pkl not found — /predict/compare will show SVM as unavailable. "
                "To enable: joblib.dump(svm_fitted, 'svm_model.pkl') in notebook.")

# Active model
if mlp_model is not None:
    model, tfidf, label_encoder, MODEL_NAME = mlp_model, mlp_tfidf, mlp_le, mlp_name
elif svm_model is not None:
    model, tfidf, label_encoder, MODEL_NAME = svm_model, svm_tfidf, svm_le, svm_name
    logger.warning("MLP unavailable — SVM activated as primary.")
else:
    model = tfidf = label_encoder = MODEL_NAME = None
    logger.error("No model loaded. Copy pkl files.")

try:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set()

_metrics = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "sentiment_counts": {"Positive": 0, "Negative": 0, "Neutral": 0},
    "low_confidence_count": 0,
}


class TextInput(BaseModel):
    text: str

    @validator("text")
    def validate_text(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("text field cannot be empty")
        if len(v) > 5000:
            raise ValueError("text exceeds 5000 character limit")
        return v

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    confidence_level: str
    scores: dict
    word_count: int
    model_used: str
    model_f1: float
    latency_ms: float
    low_confidence: bool
    request_id: str

class ExplainResponse(BaseModel):
    prediction: str
    confidence: float
    top_features: list
    model_used: str
    request_id: str

class CompareResponse(BaseModel):
    mlp: Optional[dict]
    svm: Optional[dict]
    agreement: bool
    request_id: str


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

def _conf_level(conf: float) -> str:
    if conf >= 70.0: return "high"
    if conf >= 55.0: return "moderate"
    return "low"

def _run(mdl, tv, le, text: str) -> dict:
    clean = preprocess(text)
    if not clean:
        raise ValueError("No meaningful text after preprocessing.")
    vec   = tv.transform([clean])
    proba = mdl.predict_proba(vec)[0]
    idx   = int(np.argmax(proba))
    classes = le.classes_.tolist()
    conf  = round(float(proba[idx]) * 100, 2)
    return {
        "prediction":       classes[idx],
        "confidence":       conf,
        "confidence_level": _conf_level(conf),
        "scores":           {c: round(float(p)*100, 2) for c, p in zip(classes, proba)},
        "model_used":       type(mdl).__name__,
        "model_f1":         MODEL_F1_MAP.get(type(mdl).__name__, 0.0),
        "low_confidence":   conf < 55.0,
    }


@app.on_event("startup")
async def prewarm():
    if model and tfidf:
        dummy = preprocess("meeting transcript pre warm")
        if dummy:
            tfidf.transform([dummy])
            logger.info("Model pre-warmed.")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "active_model": MODEL_NAME,
        "active_model_f1": MODEL_F1_MAP.get(MODEL_NAME or "", None),
        "primary": {"type": mlp_name, "loaded": mlp_model is not None},
        "fallback": {"type": svm_name or "SVC", "loaded": svm_model is not None},
        "version": "3.0.0",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: TextInput, request: Request):
    if model is None:
        raise HTTPException(503, "Model not loaded. Copy model.pkl/tfidf.pkl/label_encoder.pkl.")

    rid = str(uuid.uuid4())[:8]
    t0  = time.perf_counter()

    try:
        r = _run(model, tfidf, label_encoder, data.text)
        latency = round((time.perf_counter() - t0) * 1000, 2)

        _metrics["total_requests"] += 1
        _metrics["total_latency_ms"] += latency
        _metrics["sentiment_counts"][r["prediction"]] = _metrics["sentiment_counts"].get(r["prediction"], 0) + 1
        if r["low_confidence"]:
            _metrics["low_confidence_count"] += 1

        logger.info("PREDICT rid=%s pred=%s conf=%.1f%% [%s] words=%d ms=%.1f model=%s",
                    rid, r["prediction"], r["confidence"], r["confidence_level"],
                    len(data.text.split()), latency, MODEL_NAME)

        return PredictionResponse(
            **r,
            word_count=len(data.text.split()),
            latency_ms=latency,
            request_id=rid,
        )
    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Predict error rid=%s: %s", rid, exc)
        raise HTTPException(500, f"Prediction failed: {exc}")


@app.post("/predict/explain", response_model=ExplainResponse)
async def predict_explain(data: TextInput):
    """Return top-10 TF-IDF features that contributed to the prediction."""
    if model is None:
        raise HTTPException(503, "Model not loaded.")

    rid = str(uuid.uuid4())[:8]
    try:
        clean = preprocess(data.text)
        if not clean:
            raise HTTPException(400, "No meaningful text after preprocessing.")

        vec     = tfidf.transform([clean])
        proba   = model.predict_proba(vec)[0]
        idx     = int(np.argmax(proba))
        classes = label_encoder.classes_.tolist()
        pred    = classes[idx]
        conf    = round(float(proba[idx]) * 100, 2)

        vocab_inv = {v: k for k, v in tfidf.vocabulary_.items()}
        vec_arr   = vec.toarray()[0]
        nonzero   = np.where(vec_arr > 0)[0]
        scored    = sorted([(vocab_inv[i], float(vec_arr[i])) for i in nonzero],
                           key=lambda x: -x[1])[:10]

        pos_idx = list(classes).index("Positive") if "Positive" in classes else 0
        neg_idx = list(classes).index("Negative") if "Negative" in classes else 1

        top_features = []
        for word, score in scored:
            w_vec  = tfidf.transform([word])
            w_prob = model.predict_proba(w_vec)[0]
            direction = "positive" if w_prob[pos_idx] > w_prob[neg_idx] else "negative"
            top_features.append({"word": word, "tfidf_score": round(score, 4), "sentiment_direction": direction})

        return ExplainResponse(prediction=pred, confidence=conf,
                               top_features=top_features, model_used=MODEL_NAME, request_id=rid)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Explain error: %s", exc)
        raise HTTPException(500, str(exc))


@app.post("/predict/compare", response_model=CompareResponse)
async def predict_compare(data: TextInput):
    """Run MLP and SVM predictions side-by-side."""
    rid = str(uuid.uuid4())[:8]
    mlp_r = svm_r = None

    if mlp_model and mlp_tfidf:
        try:
            mlp_r = _run(mlp_model, mlp_tfidf, mlp_le, data.text)
        except Exception as e:
            mlp_r = {"error": str(e)}

    if svm_model and svm_tfidf:
        try:
            svm_r = _run(svm_model, svm_tfidf, svm_le, data.text)
        except Exception as e:
            svm_r = {"error": str(e)}

    agreement = (
        mlp_r and svm_r
        and "error" not in mlp_r and "error" not in svm_r
        and mlp_r["prediction"] == svm_r["prediction"]
    )

    return CompareResponse(mlp=mlp_r, svm=svm_r, agreement=agreement, request_id=rid)


@app.get("/metrics")
async def get_metrics():
    n = _metrics["total_requests"]
    avg = (_metrics["total_latency_ms"] / n) if n else 0
    return {
        "total_requests": n,
        "avg_latency_ms": round(avg, 2),
        "sentiment_distribution": _metrics["sentiment_counts"],
        "low_confidence_count": _metrics["low_confidence_count"],
        "low_confidence_rate_pct": round(_metrics["low_confidence_count"] / n * 100, 1) if n else 0,
    }


@app.get("/models/info")
async def models_info():
    if model is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "deployed_model": {
            "type": MODEL_NAME,
            "role": "primary (MLP)" if mlp_model else "fallback (SVM)",
            "f1_score": MODEL_F1_MAP.get(MODEL_NAME),
            "tfidf_features": tfidf.max_features,
            "tfidf_vocab_size": len(tfidf.vocabulary_),
            "classes": label_encoder.classes_.tolist(),
            "ngram_range": list(tfidf.ngram_range),
        },
        "fallback_model": {
            "type": svm_name or "SVC",
            "loaded": svm_model is not None,
            "f1_score": MODEL_F1_MAP.get(svm_name or "SVC"),
            "activate": "Copy svm_model.pkl / svm_tfidf.pkl / svm_label_encoder.pkl to enable /predict/compare",
        },
        "benchmark": [
            {"model": "Multi Layer Perceptron", "f1": 0.8805, "acc": 0.8804, "status": "PRIMARY"},
            {"model": "CNN 1D",                 "f1": 0.8600, "acc": 0.8599, "status": "TF only"},
            {"model": "Single Layer Perceptron","f1": 0.7716, "acc": 0.7714, "status": "available"},
            {"model": "SVM Linear",             "f1": 0.7596, "acc": 0.7594, "status": "FALLBACK"},
            {"model": "Logistic Regression",    "f1": 0.7465, "acc": 0.7463, "status": "available"},
            {"model": "Naive Bayes",            "f1": 0.7066, "acc": 0.7087, "status": "available"},
            {"model": "LR + SVD",               "f1": 0.5971, "acc": 0.5962, "status": "available"},
            {"model": "Decision Tree",          "f1": 0.4917, "acc": 0.5105, "status": "available"},
            {"model": "RNN",                    "f1": 0.1667, "acc": 0.3333, "status": "unstable"},
        ],
    }


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
