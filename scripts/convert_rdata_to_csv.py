import sys
import re
import pyreadr
import pandas as pd
from pathlib import Path


# ── 경로 설정 ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # scripts/ 의 상위
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"

# RData 파일명 → 저장할 CSV 파일명 매핑
FILES = {
    "TEP_FaultFree_Training.RData" : "TEP_FaultFree_Training.csv",
    "TEP_FaultFree_Testing.RData"  : "TEP_FaultFree_Testing.csv",
    "TEP_Faulty_Training.RData"    : "TEP_Faulty_Training.csv",
    "TEP_Faulty_Testing.RData"     : "TEP_Faulty_Testing.csv",
}


# ── 컬럼 정규화 ────────────────────────────────────────────────────────────────

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    RData에서 읽힌 컬럼명을 TEP 표준 형식으로 정규화합니다.

    처리하는 케이스:
      'xmeas1'   → 'xmeas_1'    (숫자 앞 언더스코어 누락)
      'XMEAS_1'  → 'xmeas_1'    (대문자)
      'xmeas.1'  → 'xmeas_1'    (R의 점 구분자)
      'xmv1'     → 'xmv_1'
      'faultNumber' / 'fault_number' / 'faultnumber' → 'faultNumber'
      'simulationRun' 등 → 'simulationRun'
    """
    new_cols = []
    for col in df.columns:
        c = col.strip().lower()

        # R 기본 구분자 점(.) → 언더스코어
        c = c.replace(".", "_")

        # 'xmeas1' → 'xmeas_1'  /  'xmv1' → 'xmv_1'
        c = re.sub(r"(xmeas)(\d+)", r"\1_\2", c)
        c = re.sub(r"(xmv)(\d+)",   r"\1_\2", c)

        # 메타 컬럼 표준화
        if c in ("faultnumber", "fault_number", "fault"):
            c = "faultNumber"
        elif c in ("simulationrun", "simulation_run", "run"):
            c = "simulationRun"
        elif c == "samplenum":
            c = "sample"

        new_cols.append(c)

    df = df.copy()
    df.columns = new_cols
    return df


# ── 단일 파일 변환 ─────────────────────────────────────────────────────────────

def convert_one(rdata_path: Path, csv_path: Path) -> pd.DataFrame:
    """
    RData 파일 1개를 읽어 CSV로 저장하고, DataFrame을 반환합니다.
    """
    print(f"\n{'─'*55}")
    print(f"  읽는 중: {rdata_path.name}")

    # 1. RData 로딩
    #    pyreadr.read_r() → OrderedDict { R객체명: DataFrame }
    try:
        result = pyreadr.read_r(str(rdata_path))
    except Exception as e:
        print(f"  [오류] RData 읽기 실패: {e}")
        sys.exit(1)

    # 2. RData 내부 객체 확인
    obj_names = list(result.keys())
    print(f"  RData 내부 객체: {obj_names}")

    # DataFrame인 객체를 찾아 사용
    df = None
    for name, obj in result.items():
        if isinstance(obj, pd.DataFrame):
            df = obj
            print(f"  사용할 객체: '{name}'  shape={obj.shape}")
            break

    if df is None:
        print(f"  [오류] DataFrame 객체를 찾을 수 없어요.")
        print(f"  내부 타입: { {k: type(v).__name__ for k, v in result.items()} }")
        sys.exit(1)

    # 3. 컬럼명 정규화
    print(f"  원본 컬럼 (앞 6개): {df.columns.tolist()[:6]} ...")
    df = normalize_columns(df)
    print(f"  정규화 후 (앞 6개): {df.columns.tolist()[:6]} ...")
    print(f"  총 컬럼 수: {len(df.columns)}")

    # 4. 결측값 확인
    n_missing = df.isnull().sum().sum()
    print(f"  결측값: {n_missing}개" + (" (정상)" if n_missing == 0 else " ← 주의!"))

    # 5. CSV 저장
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    size_kb = csv_path.stat().st_size / 1024
    print(f"  저장 완료: {csv_path.relative_to(PROJECT_ROOT)}  ({size_kb:.0f} KB)")

    return df


# ── 요약 출력 ──────────────────────────────────────────────────────────────────

def print_summary(converted: dict) -> None:
    print(f"\n{'='*55}")
    print("변환 완료 요약")
    print(f"{'='*55}")
    for csv_name, df in converted.items():
        if "faultNumber" in df.columns:
            n_normal = (df["faultNumber"] == 0).sum()
            n_fault  = (df["faultNumber"] != 0).sum()
            fault_list = sorted(df["faultNumber"].unique().tolist())
        else:
            n_normal = n_fault = "?"
            fault_list = []

        print(f"\n  [{csv_name}]")
        print(f"    rows × cols : {df.shape[0]:,} × {df.shape[1]}")
        print(f"    정상(0)     : {n_normal if isinstance(n_normal,str) else f'{n_normal:,}'}행")
        print(f"    이상(>0)    : {n_fault  if isinstance(n_fault, str) else f'{n_fault:,}'}행")
        if fault_list:
            print(f"    faultNumber : {fault_list}")


# ── 실행 ──────────────────────────────────────────────────────────────────────

def main():
    print("TEP RData → CSV 변환 시작")
    print(f"입력 : {RAW_DIR}")
    print(f"출력 : {OUT_DIR}")

    if not RAW_DIR.exists():
        print(f"\n[오류] data/raw/ 폴더가 없어요: {RAW_DIR}")
        sys.exit(1)

    # 파일 존재 여부 확인
    missing = [f for f in FILES if not (RAW_DIR / f).exists()]
    if missing:
        print(f"\n[오류] 다음 파일이 없어요:")
        for f in missing:
            print(f"  {RAW_DIR / f}")
        print(f"\ndata/raw/ 에 있는 파일:")
        for p in sorted(RAW_DIR.iterdir()):
            print(f"  {p.name}")
        sys.exit(1)

    # 변환 실행
    converted = {}
    for rdata_name, csv_name in FILES.items():
        df = convert_one(RAW_DIR / rdata_name, OUT_DIR / csv_name)
        converted[csv_name] = df

    print_summary(converted)

    print(f"\n{'='*55}")
    print("다음 단계:")
    print("  data_loader.py 또는 01_eda.ipynb 에서")
    print("  data_dir='data/processed'  로 경로를 지정해서 실행하세요.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
