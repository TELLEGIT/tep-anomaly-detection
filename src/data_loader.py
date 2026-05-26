"""
src/data_loader.py
------------------
TEP 데이터셋을 로딩하고, 학습/테스트 분리 및 라벨을 생성하는 모듈.

TEP CSV 구조:
  - faultNumber : 0 = 정상, 1~20 = 이상 유형
  - simulationRun : 독립 시뮬레이션 번호 (데이터 누수 방지를 위한 분리 기준)
  - sample        : 시뮬레이션 내 타임스텝 번호
  - xmeas_1~41   : 공정 측정값 (센서)
  - xmv_1~11     : 조작 변수 (밸브 개도 등)

주의: simulationRun 기준으로 train/test를 나눠야
      시간 순서에 의한 데이터 누수(leakage)가 없어요.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


# ── 변수 목록 정의 ────────────────────────────────────────────────────────────

# 공정 측정 센서 (xmeas_1 ~ xmeas_41)
XMEAS_COLS = [f"xmeas_{i}" for i in range(1, 42)]

# 조작 변수 (xmv_1 ~ xmv_11)
XMV_COLS = [f"xmv_{i}" for i in range(1, 12)]

# 전체 피처 컬럼 (52개)
FEATURE_COLS = XMEAS_COLS + XMV_COLS

# 메타 컬럼 (학습에 사용하지 않음)
META_COLS = ["faultNumber", "simulationRun", "sample"]

# 화공 관점 변수 그룹 (EDA 및 SHAP 해석 시 사용)
ZONE_VARS = {
    "반응기":   ["xmeas_1", "xmeas_2", "xmeas_3", "xmeas_4",
                "xmeas_7", "xmeas_8", "xmeas_9", "xmeas_21"],
    "분리기":   ["xmeas_12", "xmeas_13", "xmeas_14",
                "xmeas_15", "xmeas_16", "xmeas_17"],
    "압축기":   ["xmeas_5", "xmeas_18", "xmeas_19", "xmeas_20"],
    "제어밸브": [f"xmv_{i}" for i in range(1, 12)],
}

# 이상 유형 설명 (면접·포트폴리오 해석용)
FAULT_DESC = {
    0:  "정상 운전",
    1:  "A/C 피드 비율 이상 (Step)",
    2:  "B 성분 조성 이상 (Step)",
    3:  "D 피드 온도 이상 (Step)",
    4:  "반응기 냉각수 입구온도 이상 (Step)",
    5:  "냉각기 냉각수 입구온도 이상 (Step)",
    6:  "A 피드 손실 (Step)",
    7:  "C 헤더 압력 손실 (Step)",
    8:  "A/B/C 피드 조성 이상 (Random)",
    9:  "D 피드 온도 이상 (Random)",
    10: "C 피드 온도 이상 (Random)",
    11: "반응기 냉각수 입구온도 이상 (Random)",
    12: "냉각기 냉각수 입구온도 이상 (Random)",
    13: "반응 속도 이상 (Slow Drift)",
    14: "반응기 냉각수 밸브 고착 (Sticking)",
    15: "응축기 냉각수 밸브 고착 (Sticking)",
    16: "알 수 없음 (Unknown)",
    17: "알 수 없음 (Unknown)",
    18: "알 수 없음 (Unknown)",
    19: "알 수 없음 (Unknown)",
    20: "알 수 없음 (Unknown)",
}


# ── 메인 로딩 함수 ────────────────────────────────────────────────────────────

def load_tep(
    data_dir: str = "data/processed",   # RData 변환 후 CSV가 여기에 있어요
    fault_types: list = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    TEP CSV 4개를 로딩해 학습/테스트 DataFrame을 반환합니다.

    Parameters
    ----------
    data_dir   : CSV 파일이 있는 폴더 경로 (기본: "data/raw")
    fault_types: 사용할 이상 유형 번호 리스트 (None이면 1~20 전체 사용)
                 예: [1, 2, 4, 5]
    verbose    : True면 로딩 현황을 출력

    Returns
    -------
    train_df, test_df : label 컬럼이 추가된 DataFrame
    """
    data_dir = Path(data_dir)

    # ── CSV 파일 로딩 ────────────────────────────────────────────
    required_files = {
        "train_normal": "TEP_FaultFree_Training.csv",
        "test_normal":  "TEP_FaultFree_Testing.csv",
        "train_fault":  "TEP_Faulty_Training.csv",
        "test_fault":   "TEP_Faulty_Testing.csv",
    }

    dfs = {}
    for key, fname in required_files.items():
        fpath = data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"\n[오류] 파일을 찾을 수 없어요: {fpath}\n"
                f"  → data/raw/ 폴더에 TEP CSV 4개를 넣었는지 확인하세요.\n"
                f"  → 다운로드: https://www.kaggle.com/datasets/averkij/"
                f"tennessee-eastman-process-simulation-dataset"
            )
        dfs[key] = pd.read_csv(fpath)
        if verbose:
            print(f"  로딩 완료: {fname}  {dfs[key].shape}")

    # ── 컬럼 정규화 (소문자, 공백 제거) ─────────────────────────
    for key in dfs:
        dfs[key].columns = (
            dfs[key].columns.str.strip().str.lower().str.replace(" ", "_")
        )

    # ── 이상 유형 필터링 ─────────────────────────────────────────
    if fault_types is None:
        fault_types = list(range(1, 21))  # 전체 사용

    train_fault = dfs["train_fault"][
        dfs["train_fault"]["faultnumber"].isin(fault_types)
    ].copy()
    test_fault = dfs["test_fault"][
        dfs["test_fault"]["faultnumber"].isin(fault_types)
    ].copy()

    # ── 라벨 생성 (0=정상, 1=이상) ──────────────────────────────
    dfs["train_normal"]["label"] = 0
    dfs["test_normal"]["label"]  = 0
    train_fault["label"] = 1
    test_fault["label"]  = 1

    # ── 정상 + 이상 합치기 ───────────────────────────────────────
    train_df = pd.concat(
        [dfs["train_normal"], train_fault], ignore_index=True
    )
    test_df = pd.concat(
        [dfs["test_normal"], test_fault], ignore_index=True
    )

    # ── 컬럼명 표준화: faultnumber → faultNumber 등 ──────────────
    # (CSV에 따라 대소문자가 다를 수 있어서 통일)
    col_map = {
        "faultnumber":    "faultNumber",
        "simulationrun":  "simulationRun",
    }
    train_df.rename(columns=col_map, inplace=True)
    test_df.rename(columns=col_map, inplace=True)

    if verbose:
        _print_summary(train_df, test_df, fault_types)

    return train_df, test_df


def get_X_y(
    df: pd.DataFrame,
    feature_cols: list = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    DataFrame에서 피처 행렬(X)과 라벨 벡터(y)를 추출합니다.

    Parameters
    ----------
    df          : load_tep()로 반환된 DataFrame
    feature_cols: 사용할 피처 컬럼 리스트 (None이면 FEATURE_COLS 전체)

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)  — 0=정상, 1=이상
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # 실제 DataFrame에 있는 컬럼만 사용 (대소문자 불일치 방어)
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < len(feature_cols):
        missing = set(feature_cols) - set(available)
        print(f"  [경고] 일부 피처 컬럼이 없어요: {missing}")

    X = df[available].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    return X, y


def get_fault_subset(
    df: pd.DataFrame,
    fault_number: int,
) -> pd.DataFrame:
    """
    특정 이상 유형만 추출합니다. EDA 시 유용해요.

    Parameters
    ----------
    df           : load_tep()로 반환된 DataFrame
    fault_number : 0=정상, 1~20=이상 유형

    Returns
    -------
    subset DataFrame
    """
    return df[df["faultNumber"] == fault_number].copy()


# ── 내부 유틸 함수 ────────────────────────────────────────────────────────────

def _print_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fault_types: list,
) -> None:
    """로딩 완료 후 데이터 요약을 출력합니다."""
    print("\n" + "=" * 55)
    print("TEP 데이터 로딩 완료")
    print("=" * 55)

    for name, df in [("학습", train_df), ("테스트", test_df)]:
        n_normal = (df["label"] == 0).sum()
        n_fault  = (df["label"] == 1).sum()
        ratio    = n_fault / len(df) * 100
        print(f"\n[{name}]  총 {len(df):,}행  |  "
              f"정상: {n_normal:,}  |  이상: {n_fault:,}  "
              f"({ratio:.1f}%)")

    print(f"\n사용한 이상 유형: {fault_types}")
    print("이상 유형 설명:")
    for f in fault_types:
        print(f"  fault {f:2d}: {FAULT_DESC.get(f, '알 수 없음')}")
    print("=" * 55)


# ── 직접 실행 시 동작 확인 ────────────────────────────────────────────────────

if __name__ == "__main__":
    # 프로젝트 루트에서 python src/data_loader.py 로 실행해보세요
    train_df, test_df = load_tep(
        data_dir="data/raw",
        fault_types=[1, 2, 4, 5],
        verbose=True,
    )

    X_train, y_train = get_X_y(train_df)
    X_test,  y_test  = get_X_y(test_df)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
    print(f"y_test  shape: {y_test.shape}")
