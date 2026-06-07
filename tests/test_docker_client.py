#!/usr/bin/env python3
"""
Client test for beyondLDA2 API (Docker deployment).
Submits a bulk Si KS gap calculation and polls for results.
"""
import requests
import json
import sys
import time
import os

BASE_URL = os.environ.get("API_URL", "http://localhost:8080")

# Bulk Si input
payload = {
    "label": "docker-test-bulk-si",
    "calculation_type": "lda",
    "structure": {
        "lattice_vectors": [
            [0.0, 2.715, 2.715],
            [2.715, 0.0, 2.715],
            [2.715, 2.715, 0.0]
        ],
        "elements": ["Si", "Si"],
        "scaled_positions": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "chemical_formula": "Si2"
    },
    "parameters": {
        "xc": "LDA",
        "mode": "lcao",
        "basis": "sz(dzp)",
        "kpts": [4, 4, 4],
        "txt": "/tmp/beyondlda2-jobs/si_ks_docker.txt"
    }
}

def main():
    print(f"Testing beyondLDA2 API at {BASE_URL}")
    session = requests.Session()

    # --- Step 1: Status ---
    print("\n=== 1. Status Check ===")
    r = session.get(f"{BASE_URL}/api/v1/status")
    print(f"GET /api/v1/status → {r.status_code}")
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200, f"Status endpoint failed: {r.status_code}"

    # --- Step 2: Submit calculation ---
    print("\n=== 2. Submit Calculation ===")
    r = session.post(
        f"{BASE_URL}/api/v1/calculate",
        json=payload
    )
    print(f"POST /api/v1/calculate → {r.status_code}")
    data = r.json()
    print(json.dumps(data, indent=2))
    assert r.status_code == 202, f"Submit failed: {r.status_code}"
    job_id = data["job_id"]
    print(f"Job submitted: {job_id}")

    # --- Step 3: Poll for completion ---
    print(f"\n=== 3. Poll Job {job_id} ===")
    max_attempts = 60  # 20 minutes at 20s intervals
    for attempt in range(1, max_attempts + 1):
        time.sleep(20)
        r = session.get(f"{BASE_URL}/api/v1/jobs/{job_id}")
        status_data = r.json()
        status = status_data["status"]
        print(f"  [{attempt}] status={status}")
        if status == "completed":
            print("\n✅ Job completed!")
            print(json.dumps(status_data, indent=2))
            return status_data
        elif status == "failed":
            print(f"\n❌ Job failed: {status_data.get('error', 'unknown error')}")
            print(json.dumps(status_data, indent=2))
            sys.exit(1)
        # else: still queued/running

    print(f"\n⏰ Timeout after {max_attempts * 20}s")
    r = session.get(f"{BASE_URL}/api/v1/jobs/{job_id}")
    print(json.dumps(r.json(), indent=2))
    sys.exit(1)


if __name__ == "__main__":
    result = main()
    print("\n=== Summary ===")
    ks_gap = result.get("result", {}).get("ks_gap")
    if ks_gap:
        print(f"Si KS band gap (LDA): {ks_gap:.3f} eV")
    if result.get("result", {}).get("output_files"):
        print("Output files:")
        for f in result["result"]["output_files"]:
            print(f"  - {f}")
