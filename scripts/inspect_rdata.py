"""
scripts/inspect_rdata.py
-------------------------
RData 파일 내부 구조를 먼저 확인하는 디버깅 스크립트.
변환 전에 한 번 실행해서 컬럼명·데이터 타입을 파악하세요.

실행 방법:
    python scripts/inspect_rdata.py
"""
import sys
from pathlib import Path

try:
    import pyreadr
except ImportError:
    print("[오류] pip install pyreadr")
    sys.exit(1)

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
FILES = [
    "TEP_FaultFree_Training.RData",
    "TEP_FaultFree_Testing.RData",
    "TEP_Faulty_Training.RData",
    "TEP_Faulty_Testing.RData",
]

print("=" * 60)
print("RData 파일 내부 구조 확인")
print("=" * 60)

for fname in FILES:
    fpath = RAW_DIR / fname
    if not fpath.exists():
        print(f"\n[없음] {fname}")
        continue

    print(f"\n── {fname}")
    try:
        result = pyreadr.read_r(str(fpath))
        keys = list(result.keys())
        print(f"   R 객체 이름: {keys}")

        df = result[keys[0]]
        print(f"   shape      : {df.shape}")
        print(f"   컬럼 (전체): {df.columns.tolist()}")
        print(f"   dtypes     :")
        print(df.dtypes.to_string(header=False).replace("^", "     "))
        print(f"   head(2)    :")
        print(df.head(2).to_string())

    except Exception as e:
        print(f"   [오류] {e}")

print("\n" + "=" * 60)
print("위 출력을 확인 후 convert_rdata_to_csv.py 를 실행하세요.")
