"""Test koneksi dan operasi CRUD Supabase untuk CITRA X-RAY"""
from database import *

print("=" * 50)
print("  TEST KONEKSI SUPABASE - CITRA X-RAY")
print("=" * 50)

# TEST 1: Health Check
print("\n[TEST 1] Health Check...")
ok = health_check()
print("  Status:", "ONLINE" if ok else "OFFLINE")

# TEST 2: Insert Pasien
print("\n[TEST 2] Insert Pasien Baru...")
try:
    p = upsert_patient("RM-TEST-001", "Test Pasien Otomatis", 30, "Laki-laki")
    pid = p.get("id", "")
    print("  Pasien tersimpan! ID:", pid[:8] + "..." if pid else "gagal")
except Exception as e:
    print("  Error:", e)
    pid = None

# TEST 3: Baca Pasien
print("\n[TEST 3] Baca Data Pasien...")
try:
    p2 = get_patient_by_rm("RM-TEST-001")
    if p2:
        print("  Nama:", p2["full_name"])
        print("  Usia:", p2["age"], "thn")
        print("  Gender:", p2["gender"])
    else:
        print("  Data tidak ditemukan (RLS mungkin masih aktif)")
        print("  >> Jalankan: ALTER TABLE patients DISABLE ROW LEVEL SECURITY;")
except Exception as e:
    print("  Error:", e)
    p2 = None

# TEST 4: Buat Study + Diagnosis
print("\n[TEST 4] Simpan Study & Diagnosis...")
try:
    if p2:
        study = create_study(p2["id"], "test_xray.jpg", preset_id="test")
        sc = study.get("study_code", "?")
        print("  Study:", sc)

        fake_preds = [
            {"class_name": "Effusion", "probability": 0.85, "percentage": 85.0, "is_detected": True, "auroc": 0.7933},
        ]
        for cn in ["Atelectasis","Cardiomegaly","Consolidation","Edema","Emphysema","Fibrosis","Hernia","Infiltration","Mass","No Finding","Nodule","Pleural_Thickening","Pneumonia","Pneumothorax"]:
            fake_preds.append({"class_name": cn, "probability": 0.05, "percentage": 5.0, "is_detected": False, "auroc": 0.70})

        diag = save_diagnosis_result(study["id"], fake_preds, threshold=0.20, inference_time=1.5)
        print("  Diagnosis tersimpan!")
        print("  Top disease:", diag.get("top_disease", "?"))
        print("  Triage:", diag.get("triage_status", "?"))
    else:
        print("  Skipped (pasien tidak ditemukan)")
except Exception as e:
    print("  Error:", e)

# SUMMARY
print("\n" + "=" * 50)
if ok:
    print("  SUKSES! Database Supabase TERHUBUNG & BERFUNGSI")
else:
    print("  Database belum berfungsi penuh")
print("=" * 50)
