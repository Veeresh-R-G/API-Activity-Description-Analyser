# ============================================================
# 🏃 Workout Coach API
# FastAPI endpoint powered by fine-tuned DistilBERT
# Classifies workout effort + generates coaching advice
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
import uvicorn
from datetime import datetime

# ── App Setup ─────────────────────────────────────────────
app = FastAPI(
    title="🏃 Workout Coach API",
    description="DistilBERT-powered workout effort classifier + coaching engine",
    version="1.0.0"
)

# Allow all origins for demo (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model Once at Startup ─────────────────────────────
MODEL_PATH = "veeresh11/workout-activity-classifier"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading DistilBERT from {MODEL_PATH}...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model = model.to(device)
model.eval()
print(f"✅ Model loaded on {device}")

# ── Constants ──────────────────────────────────────────────
EFFORT_LABELS = {
    0: "Low Effort 🟢",
    1: "Moderate Effort 🟡",
    2: "High Effort 🔴"
}

# ── Request + Response Models ──────────────────────────────


class WorkoutRequest(BaseModel):
    description: str = Field(
        ...,
        example="Brutal interval session. Legs on fire.",
        description="Workout name or description from Strava"
    )
    avg_hr: Optional[float] = Field(
        None, ge=50, le=220,
        description="Average heart rate in BPM"
    )
    distance_km: Optional[float] = Field(
        None, ge=0,
        description="Distance covered in km"
    )
    duration_min: Optional[float] = Field(
        None, ge=0,
        description="Duration in minutes"
    )
    suffer_score: Optional[int] = Field(
        None, ge=0, le=1000,
        description="Strava suffer score"
    )
    acwr: Optional[float] = Field(
        None, ge=0, le=5,
        description="Acute:Chronic Workload Ratio (from Project 3)"
    )


class WorkoutResponse(BaseModel):
    effort_level: str
    effort_code: int
    confidence: float
    coaching_advice: str
    recovery_rec: str
    next_session_rec: str
    warning: Optional[str]
    timestamp: str

# ── Coaching Engine ────────────────────────────────────────


def generate_coaching_advice(
    effort_code: int,
    avg_hr: Optional[float],
    distance_km: Optional[float],
    suffer_score: Optional[int],
    acwr: Optional[float]
) -> dict:
    """
    Generates personalised coaching advice combining:
    - NLP effort classification
    - Physiological data (HR, distance, suffer score)
    - Training load context (ACWR from Project 3)

    This is the coaching intelligence layer —
    what separates a classifier from a coaching product.
    """
    advice = ""
    recovery_rec = ""
    next_session = ""
    warning = None

    # ── High Effort ──────────────────────────────────────
    if effort_code == 2:
        advice = (
            "Great high-intensity session! Your body has been "
            "stressed — now recovery becomes the training."
        )
        recovery_rec = (
            "Take 24-48 hours of easy movement before your "
            "next hard session. Prioritise sleep and nutrition."
        )
        next_session = (
            "Next session: easy 20-30 min recovery jog, "
            "HR below 140 BPM. No intervals for 48 hours."
        )

        # Enrich with HR data if available
        if avg_hr and avg_hr > 175:
            advice += (
                f" Your average HR of {avg_hr:.0f} BPM indicates "
                f"you worked near your maximum — excellent effort."
            )

        # ACWR warning
        if acwr and acwr > 1.3:
            warning = (
                f"⚠️ Your ACWR is {acwr:.2f} — above the safe zone (0.8–1.3). "
                f"Consider an extra rest day to prevent overtraining injury."
            )

    # ── Moderate Effort ──────────────────────────────────
    elif effort_code == 1:
        advice = (
            "Solid moderate session — this is where aerobic "
            "fitness is built. Consistent moderate effort "
            "is the backbone of endurance training."
        )
        recovery_rec = (
            "Standard 12-24 hour recovery. Light stretching "
            "and good hydration. You're in good shape."
        )
        next_session = (
            "Next session: you can either repeat moderate effort "
            "or step up to a quality session — body is ready."
        )

        if avg_hr and distance_km:
            pace = distance_km / (avg_hr / 60) if avg_hr > 0 else 0
            advice += (
                f" Covering {distance_km:.1f}km at "
                f"{avg_hr:.0f} BPM avg HR shows good aerobic efficiency."
            )

    # ── Low Effort ───────────────────────────────────────
    else:
        advice = (
            "Easy session logged — this is not wasted training. "
            "Easy miles build aerobic base and accelerate recovery "
            "from harder sessions. This is smart training."
        )
        recovery_rec = (
            "Minimal recovery needed. You can train again "
            "tomorrow with confidence. Body is fresh."
        )
        next_session = (
            "Next session: perfect time for a quality workout. "
            "You're recovered — make it count with intervals "
            "or a tempo run."
        )

        if suffer_score and suffer_score < 10:
            advice += (
                " Your suffer score confirms this was a true "
                "recovery session — exactly what was planned."
            )

    return {
        "coaching_advice": advice,
        "recovery_rec": recovery_rec,
        "next_session_rec": next_session,
        "warning": warning
    }

# ── Inference Function ─────────────────────────────────────


def classify_effort(text: str):
    """Run DistilBERT inference on workout description"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = probs.argmax().item()

    return pred, float(probs[pred])

# ── API Endpoints ──────────────────────────────────────────


@app.get("/")
async def root():
    return {
        "message": "🏃 Workout Coach API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Classify workout effort + get coaching advice",
            "POST /batch": "Analyze multiple workouts at once",
            "GET  /health": "API health check",
            "GET  /docs": "Interactive API documentation"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "DistilBERT fine-tuned",
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze", response_model=WorkoutResponse)
async def analyze_workout(request: WorkoutRequest):
    """
    Main endpoint: classify workout effort + generate coaching advice.

    Combines:
    - DistilBERT NLP classification of description text
    - Physiological context (HR, distance, suffer score)
    - Training load context (ACWR)

    Returns effort classification + personalised coaching advice.
    """
    if not request.description.strip():
        raise HTTPException(
            status_code=400,
            detail="Description cannot be empty"
        )

    # Classify effort from text
    effort_code, confidence = classify_effort(request.description)

    # Generate coaching advice
    coaching = generate_coaching_advice(
        effort_code=effort_code,
        avg_hr=request.avg_hr,
        distance_km=request.distance_km,
        suffer_score=request.suffer_score,
        acwr=request.acwr
    )

    return WorkoutResponse(
        effort_level=EFFORT_LABELS[effort_code],
        effort_code=effort_code,
        confidence=round(confidence, 4),
        coaching_advice=coaching["coaching_advice"],
        recovery_rec=coaching["recovery_rec"],
        next_session_rec=coaching["next_session_rec"],
        warning=coaching["warning"],
        timestamp=datetime.now().isoformat()
    )


@app.post("/batch")
async def batch_analyze(requests: list[WorkoutRequest]):
    """
    Analyze multiple workouts at once.
    Useful for processing full Strava history.
    """
    if len(requests) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 workouts per batch request"
        )

    results = []
    for req in requests:
        effort_code, confidence = classify_effort(req.description)
        coaching = generate_coaching_advice(
            effort_code=effort_code,
            avg_hr=req.avg_hr,
            distance_km=req.distance_km,
            suffer_score=req.suffer_score,
            acwr=req.acwr
        )
        results.append({
            "description": req.description,
            "effort_level": EFFORT_LABELS[effort_code],
            "effort_code": effort_code,
            "confidence": round(confidence, 4),
            "advice": coaching["coaching_advice"],
            "warning": coaching["warning"]
        })

    return {
        "total": len(results),
        "results": results,
        "summary": {
            "high_effort_sessions": sum(1 for r in results if r["effort_code"] == 2),
            "moderate_effort_sessions": sum(1 for r in results if r["effort_code"] == 1),
            "low_effort_sessions": sum(1 for r in results if r["effort_code"] == 0),
            "warnings": sum(1 for r in results if r["warning"])
        }
    }

# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
