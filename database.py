"""
database.py
Supabase Database Integration Layer for ThoraxVision PACS-AI
Multi-Label Thoracic Disease Classification (DenseNet121+GRU)

Handles:
- Patient records management
- X-Ray study sessions
- Diagnosis results (15-class predictions)
- Score-CAM outputs
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

_supabase_client = None

def get_supabase():
    """Lazy-init Supabase client (singleton)."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise ValueError(
            "Supabase credentials belum dikonfigurasi. "
            "Salin file .env.example ke .env dan isi SUPABASE_URL & SUPABASE_ANON_KEY."
        )
    try:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        print("[Supabase] Koneksi database berhasil.")
        return _supabase_client
    except Exception as e:
        print(f"[Supabase] Gagal koneksi: {e}")
        raise


# ---------------------------------------------------------------------------
# PATIENTS
# ---------------------------------------------------------------------------

def upsert_patient(rm_number: str, full_name: str, age: Optional[int] = None,
                   gender: Optional[str] = None) -> dict:
    """Insert or update a patient record. Returns the patient row."""
    sb = get_supabase()
    data = {
        "rm_number": rm_number,
        "full_name": full_name,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    if age:
        data["age"] = age
    if gender:
        data["gender"] = gender

    res = sb.table("patients").upsert(data, on_conflict="rm_number").execute()
    return res.data[0] if res.data else {}


def get_patient_by_rm(rm_number: str) -> Optional[dict]:
    """Fetch a patient by RM number."""
    sb = get_supabase()
    res = sb.table("patients").select("*").eq("rm_number", rm_number).single().execute()
    return res.data


# ---------------------------------------------------------------------------
# X-RAY STUDIES
# ---------------------------------------------------------------------------

def create_study(patient_id: str, image_filename: str,
                 preset_id: Optional[str] = None,
                 image_base64: Optional[str] = None,
                 wl_mode: str = "default") -> dict:
    """Create a new X-Ray study session for a patient."""
    sb = get_supabase()
    study_code = f"STD-{uuid.uuid4().hex[:8].upper()}"
    data = {
        "patient_id": patient_id,
        "study_code": study_code,
        "image_filename": image_filename,
        "preset_id": preset_id,
        "wl_mode": wl_mode,
        "acquisition_time": datetime.now(timezone.utc).isoformat(),
    }
    if image_base64:
        # Only store first 500 chars as reference; full base64 is large
        data["image_base64"] = image_base64[:500] + "...[truncated]"

    res = sb.table("xray_studies").insert(data).execute()
    return res.data[0] if res.data else {}


# ---------------------------------------------------------------------------
# DIAGNOSIS RESULTS
# ---------------------------------------------------------------------------

_CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

_PROB_FIELD_MAP = {
    'Atelectasis':        'prob_atelectasis',
    'Cardiomegaly':       'prob_cardiomegaly',
    'Consolidation':      'prob_consolidation',
    'Edema':              'prob_edema',
    'Effusion':           'prob_effusion',
    'Emphysema':          'prob_emphysema',
    'Fibrosis':           'prob_fibrosis',
    'Hernia':             'prob_hernia',
    'Infiltration':       'prob_infiltration',
    'Mass':               'prob_mass',
    'No Finding':         'prob_no_finding',
    'Nodule':             'prob_nodule',
    'Pleural_Thickening': 'prob_pleural_thickening',
    'Pneumonia':          'prob_pneumonia',
    'Pneumothorax':       'prob_pneumothorax',
}

_DET_FIELD_MAP = {
    'Atelectasis':        'det_atelectasis',
    'Cardiomegaly':       'det_cardiomegaly',
    'Consolidation':      'det_consolidation',
    'Edema':              'det_edema',
    'Effusion':           'det_effusion',
    'Emphysema':          'det_emphysema',
    'Fibrosis':           'det_fibrosis',
    'Hernia':             'det_hernia',
    'Infiltration':       'det_infiltration',
    'Mass':               'det_mass',
    'No Finding':         'det_no_finding',
    'Nodule':             'det_nodule',
    'Pleural_Thickening': 'det_pleural_thickening',
    'Pneumonia':          'det_pneumonia',
    'Pneumothorax':       'det_pneumothorax',
}


def save_diagnosis_result(study_id: str, predictions: list, threshold: float,
                          inference_time: float = 0.0,
                          finding_cor: str = "", finding_pulmo: str = "",
                          finding_pleura: str = "", finding_bones: str = "",
                          conclusion: str = "") -> dict:
    """
    Save 15-class prediction results to Supabase.
    
    Args:
        study_id: UUID of the xray_studies row
        predictions: list of dicts [{class_name, probability, percentage, auroc, is_detected}, ...]
        threshold: classification cutoff used
        inference_time: seconds taken by model inference
        finding_cor / pulmo / pleura / bones / conclusion: Radiology expertise sheet fields
    
    Returns:
        Saved diagnosis_results row dict
    """
    sb = get_supabase()

    detected = [p for p in predictions if p.get("is_detected") or p.get("probability", 0) >= threshold]
    top_pred = max(predictions, key=lambda x: x["probability"])
    triage = "alert" if detected else "normal"

    data = {
        "study_id": study_id,
        "threshold_used": round(threshold, 2),
        "model_architecture": "DenseNet121+GRU",
        "mean_auroc": 0.7111,
        "inference_time_sec": round(inference_time, 3),
        "total_detected": len(detected),
        "top_disease": top_pred["class_name"],
        "top_probability": round(top_pred["probability"] * 100, 2),
        "triage_status": triage,
        "clinical_impression": f"{len(detected)} patologi terdeteksi" if detected else "Dalam batas normal",
        "finding_cor": finding_cor,
        "finding_pulmo": finding_pulmo,
        "finding_pleura": finding_pleura,
        "finding_bones": finding_bones,
        "conclusion": conclusion,
    }

    # Map per-class probabilities and detection flags
    for pred in predictions:
        name = pred["class_name"]
        prob = round(float(pred["probability"]), 4)
        is_det = prob >= threshold

        if name in _PROB_FIELD_MAP:
            data[_PROB_FIELD_MAP[name]] = prob
        if name in _DET_FIELD_MAP:
            data[_DET_FIELD_MAP[name]] = is_det

    res = sb.table("diagnosis_results").insert(data).execute()
    return res.data[0] if res.data else {}


# ---------------------------------------------------------------------------
# SCORE-CAM OUTPUTS
# ---------------------------------------------------------------------------

def save_scorecam_output(result_id: str, disease_class: str, class_index: int,
                         probability: float, auroc_score: float,
                         alpha_blend: float = 0.45, top_k: int = 20,
                         original_url: str = "", heatmap_url: str = "",
                         overlay_url: str = "") -> dict:
    """Save a Score-CAM visualization result to Supabase."""
    sb = get_supabase()
    data = {
        "result_id": result_id,
        "disease_class": disease_class,
        "class_index": class_index,
        "probability": round(float(probability), 4),
        "auroc_score": round(float(auroc_score), 4),
        "alpha_blend": round(float(alpha_blend), 2),
        "top_k": top_k,
        "original_image_url": original_url,
        "heatmap_image_url": heatmap_url,
        "overlay_image_url": overlay_url,
    }
    res = sb.table("scorecam_outputs").insert(data).execute()
    return res.data[0] if res.data else {}


# ---------------------------------------------------------------------------
# HISTORY / QUERY
# ---------------------------------------------------------------------------

def get_recent_studies(limit: int = 20) -> list:
    """Fetch recent diagnosis history via the v_study_summary view."""
    sb = get_supabase()
    res = sb.table("v_study_summary").select("*").limit(limit).execute()
    return res.data or []


def get_study_detail(result_id: str) -> Optional[dict]:
    """Get a full diagnosis result by result_id."""
    sb = get_supabase()
    res = sb.table("diagnosis_results").select("*, xray_studies(*, patients(*))").eq("id", result_id).single().execute()
    return res.data


def get_scorecam_for_result(result_id: str) -> list:
    """Get all Score-CAM outputs for a given diagnosis result."""
    sb = get_supabase()
    res = sb.table("scorecam_outputs").select("*").eq("result_id", result_id).execute()
    return res.data or []


def health_check() -> bool:
    """Check if Supabase connection is alive."""
    try:
        sb = get_supabase()
        sb.table("patients").select("id").limit(1).execute()
        return True
    except Exception:
        return False


if __name__ == "__main__":
    print("=== ThoraxVision Supabase Health Check ===")
    ok = health_check()
    print(f"[OK] Supabase connection: {'ONLINE' if ok else 'OFFLINE / Not configured'}")
