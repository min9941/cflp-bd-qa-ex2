# Hybrid Quantum–Classical Benders Decomposition for MS-CFLP

> **Research code · Python · Gurobi · D-Wave Ocean**  
> Capacitated Facility Location Problem을 Benders decomposition으로 분해하고, Master Problem을 QUBO로 변환해 Simulated Annealing(SA) 및 Quantum Annealing(QA)으로 탐색하는 실험 프레임워크.

이 저장소(`cflp-bd-qa-ex2`)는 **Multiple-Source Capacitated Facility Location Problem(MS-CFLP)** 을 대상으로 다음을 분석한다.

- **Solution quality:** Gurobi 기준해와 SA/QA 기반 Benders의 최종 해, feasibility 및 optimality gap
- **QUBO complexity:** Benders optimality cut 누적에 따른 logical variables/edges, density, coefficient range, slack bits
- **Minor embedding:** Pegasus와 Zephyr에서의 임베딩 성공률, 소요 시간, physical qubits 및 chain lengths
- **Theta-bound analysis:** 저장된 cut trajectory를 재사용한 $\theta$ 구간 및 정밀도 변화의 **offline QUBO 구조 분석**

> [!IMPORTANT]
> **현재 구현 기준**: 기본 Benders 실행은 모든 cut을 누적하고, 각 iteration의 임베딩은 `minorminer`로 **독립적으로 다시 탐색**한다. `initial_chains` 기반 재사용, cut selection, reverse annealing, facility-level solution correction은 아직 기본 실행 경로에 구현되어 있지 않다. 또한 theta-bound offline 분석은 **실제 QA 성능 향상을 입증하지 않는다.**

## Contents

1. [Quick start](#quick-start)
2. [Formulation & algorithm](#formulation--algorithm)
3. [QUBO encoding](#qubo-encoding)
4. [Experiments](#experiments)
5. [Theta-bound offline analysis](#theta-bound-offline-analysis)
6. [Configuration](#configuration)
7. [Repository structure](#repository-structure)
8. [Outputs & reproducibility](#outputs--reproducibility)
9. [Validation & limitations](#validation--limitations)

---

## Quick start

### 1. Requirements

- **Python 3.10+**
- **Gurobi** (`gurobipy`) 및 유효한 라이선스: 원문제, Benders MP/SP 및 일부 검증에 필요
- Python 패키지: `requirements.txt` 참조 (`numpy`, `pandas`, `pyyaml`, `networkx`, `dimod`, `dwave-neal`, `minorminer`, `dwave-system` 등)
- **D-Wave Leap 계정 및 QPU 접근 권한**: 실제 QA 실행에만 필요. 이상적 topology의 embedding 실험에는 QPU 접근이 필요하지 않음

저장소 루트에서 실행한다.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install -e .
```

> `pip install`은 Gurobi 최적화 라이선스나 D-Wave QPU 사용 권한을 제공하지 않는다. `config/base.yaml`의 기본 `qa.profile: lab_account`는 특정 환경의 프로필 이름이므로, 자신의 환경에 맞게 수정해야 한다. **API 토큰을 YAML 또는 Git에 기록하지 말 것.**

### 2. Generate data and run the classical baseline

```bash
# pilot.yaml에 정의된 생성 인스턴스 만들기
python scripts/generate_data.py

# 원문제 Gurobi + Gurobi-controlled Benders (theta discretization 비교 포함)
python scripts/run_gurobi.py --instances 4x12_s100 --encoded-mp

# SA가 MP 후보를 만드는 end-to-end Benders
python scripts/run_sa.py --instances 4x12_s100
```

기본 설정의 `pilot.yaml`은 `4x12`, `8x25`, `16x50`, `25x50` 각각 5개 seed를 정의한다. `--instances`는 실행할 **instance ID**를 쉼표로 구분하며, `generate_data.py`에는 이 옵션이 없다.

### 3. Embedding study (without submitting to a QPU)

```bash
# 이상적 Pegasus P16 / Zephyr Z12에 임베딩
python scripts/run_embedding.py --instances 4x12_s100

# 실제 접근 가능한 QPU working graph까지 포함 (접근 권한 필요)
python scripts/run_embedding.py --instances 4x12_s100 --include-actual

# 실제 working graph만 시험
python scripts/run_embedding.py --instances 4x12_s100 --actual-only
```

`run_embedding.py`는 **Gurobi-controlled cut trajectory**의 QUBO snapshot을 임베딩한다. 실행 결과를 QA 샘플링 결과와 혼동하지 않는다.

### 4. Quantum annealing (optional)

D-Wave 인증과 QPU 접근을 설정한 후 실행한다.

```bash
# PowerShell: $env:DWAVE_API_TOKEN = "<YOUR_TOKEN>"
# macOS / Linux: export DWAVE_API_TOKEN="<YOUR_TOKEN>"

python scripts/run_qa.py --instances 4x12_s100 --topology pegasus
python scripts/run_qa.py --instances 4x12_s100 --topology zephyr
```

`run_qa.py`는 실제 working graph를 확인하고 embedding에 성공한 seed들을 QPU 평가에 사용한다. **이상적 Pegasus/Zephyr graph에 QPU 샘플을 제출하지 않는다.** QPU 인증 실패·할당량 소진 등은 구체적인 status로 기록한다.

### 5. Tests

```bash
python -m pytest -q
```

테스트 중 일부는 `gurobipy`·D-Wave 패키지와 실행 환경에 의존할 수 있다. 실제 QPU 재현성까지 이 명령 하나로 보장되는 것은 아니다.

---

## Formulation & algorithm

### MS-CFLP

시설 집합 $J$, 고객 집합 $I$에 대해 다음 변수를 사용한다.

| Symbol | Meaning |
|---|---|
| $y_j\in\{0,1\}$ | 시설 $j$의 개설 여부 |
| $x_{ij}\ge 0$ | 고객 $i$의 수요 중 시설 $j$에 배정하는 비율 |
| $f_j$ | 시설 $j$의 고정 개설 비용 |
| $s_j$ | 시설 $j$의 용량 |
| $d_i$ | 고객 $i$의 수요 |
| $c_{ij}$ | 단위 수요당 운송 비용 |

$$
\begin{aligned}
\min_{x,y}\quad &\sum_{j\in J}f_jy_j+\sum_{i\in I}\sum_{j\in J}d_ic_{ij}x_{ij}\\
\text{s.t.}\quad &\sum_{j\in J}x_{ij}=1 && \forall i\in I,\\
&\sum_{i\in I}d_ix_{ij}\le s_jy_j &&\forall j\in J,\\
&y_j\in\{0,1\},\quad x_{ij}\ge 0.
\end{aligned}
$$

고객 수요를 여러 시설에 나누어 배정할 수 있는 **multiple-source** 모형이다. 별도의 linking constraint $x_{ij}\le y_j$는 두지 않는다. $d_i>0$이므로 용량 제약만으로 폐쇄 시설의 flow가 0이 된다.

### Benders decomposition

총수요 $D=\sum_i d_i$와 recourse variable $\theta$를 사용한다. Master Problem(MP)은 시설 개설을 결정하고, Subproblem(SP)은 주어진 $\bar y$에 대한 운송 최적값 $Q(\bar y)$를 계산한다.

**Master Problem**

$$
\begin{aligned}
\min_{y,\theta}\quad &\sum_jf_jy_j+\theta\\
\text{s.t.}\quad &\sum_js_jy_j\ge D,\\
&\theta+\sum_js_jv_j^ky_j\ge\sum_iu_i^k &&\forall k\in\mathcal K,\\
&y_j\in\{0,1\},\quad\theta\ge0.
\end{aligned}
$$

**Subproblem의 dual** (주어진 $\bar y$)

$$
\begin{aligned}
\max_{u,v}\quad &\sum_i u_i-\sum_js_j\bar y_jv_j\\
\text{s.t.}\quad &u_i-d_iv_j\le d_ic_{ij}&&\forall i,j,\\
&u_i\text{ free},\quad v_j\ge0.
\end{aligned}
$$

SP dual solution $(u^k,v^k)$에서 **single optimality cut**을 생성한다.

$$
\boxed{\theta+\sum_js_jv_j^ky_j\ge\sum_i u_i^k}
$$

모든 고객–시설 arc가 존재하고 배정 변수가 연속이므로, aggregate capacity constraint $\sum_j s_j y_j\ge D$를 만족하면 SP는 이론적으로 feasible하다. 따라서 별도의 feasibility cut을 생성하지 않으며, SP가 infeasible이면 진단 status와 함께 중단한다.

### One Benders iteration

```text
                ┌──────────────────────────────┐
                │ Active Benders optimality cuts│
                └──────────────┬───────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
      Gurobi continuous-θ MP             Candidate MP solver
      (binary y, real θ)                 ├─ Gurobi (control)
              │                         ├─ SA (QUBO)
              │                         └─ QA (QUBO + embedding)
      Certified lower bound                      │
                                                ▼
                                      Candidate facility vector y
                                                │
                                                ▼
                                        Gurobi transportation SP
                                                │
                       ┌────────────────────────┴───────────────┐
                       ▼                                        ▼
                 Feasible objective                      Dual solution
                 / upper bound                        (u, v) / new cut
                       │                                        │
                       └──────────────────┬─────────────────────┘
                                          ▼
                                  Check gap / status
                                          │
                               Next iteration if needed
```

**중요:** `continuous_mp`의 $y$는 **binary**이고 $\theta$만 continuous이다. SA/QA가 만든 objective를 certified lower bound로 사용하지 않는다. 해당 iteration의 **Gurobi continuous-θ MP 최적값**에서 lower bound를 얻고, 후보 $y$의 SP 평가로 feasible upper bound를 갱신한다. 직접 푼 original CFLP의 objective는 `OPTIMAL`일 때만 최적값(ground truth)으로 취급한다.

`models/master.py`는 `continuous_mp` 및 (선택적인) `encoded_mp`를 구분한다. `encoded_mp`는 $\theta$만 binary로 표현하면서 원래 inequality를 보존하므로 penalty/slack 영향과 discretization 영향을 구분할 수 있다.

---

## QUBO encoding

### Objective and penalty

$$
H(z)=\underbrace{\sum_j f_jy_j+\theta(t)}_{H_{\mathrm{obj}}}
+\alpha_{\mathrm{cap}}\,r_{\mathrm{cap}}^2
+\sum_{k\in\mathcal K}\alpha_k\,r_k^2,
$$

$$
\begin{aligned}
r_{\mathrm{cap}}&=\sum_js_jy_j-D-q_{\mathrm{cap}},\\
r_k&=\theta(t)+\sum_js_jv_j^ky_j-\sum_i u_i^k-q_k.
\end{aligned}
$$

현재 baseline의 penalty는 모든 제약에 동일한 규칙 $\alpha=1.1U_{\mathrm{obj}}$를 사용한다(`config/base.yaml`의 `penalty.margin`). 이는 임베딩된 물리 큐비트를 연결하는 **chain strength**와 다른 개념이다. Chain strength는 QA 단계에서 `uniform_torque_compensation`으로 계산한다.

### Binary encoding

$$
\theta=\theta_L+\delta_\theta\sum_{p=0}^{b_\theta-1}2^pt_p,
\qquad
\delta_\theta=\frac{\theta_U-\theta_L}{2^{b_\theta}-1}.
$$

현재 기본 실행은 $b_\theta=10$, $\theta_L=0$, $\theta_U=U_{\mathrm{obj}}$를 사용한다.

$$
U_{\mathrm{obj}}=\sum_j f_j+\sum_i\sum_jd_ic_{ij}.
$$

- **Capacity slack**: $q_{\mathrm{cap}}\in[0,\sum_j s_j-D]$, 정수 간격 $\delta_{\mathrm{cap}}=1$.
- **Cut slack**: 각 cut의 $q_k$를 별도 binary encoding하며, $\delta_{\mathrm{cut}}=0.25\delta_\theta$를 사용한다. `0.25`는 현 baseline의 실험 설정이지 이론적 최적값이 아니다.
- **Bit budget**: slack bit 수는 표현 범위와 간격에 따라 자동 계산한다. 마지막 bit를 줄이는 truncated encoding은 사용하지 않는다.
- **Feasibility**: cut dual 계수는 실수이므로, 원래 MP inequality를 만족해도 QUBO equality residual이 정확히 0일 필요는 없다. Sample 선택은 디코딩한 $y,\theta$의 **원래 MP 제약 위반량**에 기반한다.

구현: `encoding/`, `qubo/builder.py`, `qubo/decode.py`.

---

## Experiments

### Execution matrix

| Experiment | MP candidate / trajectory | QPU needed? | Entry point |
|---|---|---|---|
| Original CFLP & controlled Benders | Gurobi | No | `scripts/run_gurobi.py` |
| SA end-to-end Benders | SA/QUBO | No | `scripts/run_sa.py` |
| Structural controlled embedding | Gurobi cut snapshots | No (ideal); access required (actual) | `scripts/run_embedding.py` |
| QA end-to-end Benders | QA/QUBO | Yes | `scripts/run_qa.py` |
| Full sequential pilot | Above steps | Optional with flags | `scripts/run_pilot.py` |
| QUBO construction / exact validation | Saved trajectories | No | `notebooks/03_build_and_validate_qubo.ipynb` |
| Offline theta-bound comparison | Saved Gurobi/SA/QA cuts | No QPU; S3 requires Gurobi LP | `notebooks/10_1`–`10_5` |

실험을 한 번에 호출할 수도 있다.

```bash
# QA와 실제 QPU working-graph 탐색은 제외한 순차 파이프라인
python scripts/run_pilot.py --instances 4x12_s100 --skip-qa --skip-actual-embedding
```

`run_pilot.py`는 데이터 생성 → Gurobi → SA → ideal embedding → (선택) actual embedding → (선택) Pegasus/Zephyr QA 순서다. **앞 단계가 실패해도 뒤 단계를 시도**하므로 마지막 단계별 return code를 확인해야 한다. 생성 단계에는 `--instances`가 전달되지 않으며, 전체 기본 인스턴스를 생성한다.

### Available data

| Directory | Content |
|---|---|
| `data/raw/` | 설정과 seed로 생성하는 MS-CFLP JSON (`4x12`, `8x25`, `16x50`, `25x50`) |
| `data/benchmark/raw_or_library/` | OR-Library `cap*.txt` 원본 |
| `data/benchmark/` | 변환된 benchmark JSON (`16x50_b41`–`b44`, `16x50_b51`, `25x50_b81`–`b84`, `25x50_b91`) |

파일명 `25x50_b81`은 **시설 25개, 고객 50개**의 `cap81` benchmark를 의미한다. Benchmark는 무작위 생성 데이터와 다르며 seed 대신 구분용 값 `-1`을 사용한다.

```bash
# 포함된 OR-Library 텍스트를 JSON으로 다시 변환
python scripts/convert_benchmark.py data/benchmark/raw_or_library

# benchmark 하나로 Gurobi/SA 실행 (지원되는 --data 옵션)
python scripts/run_gurobi.py --data data/benchmark --instances 25x50_b81
python scripts/run_sa.py --data data/benchmark --instances 25x50_b81
```

### Notebooks

| Notebook | Primary purpose |
|---|---|
| `01_generated_instances.ipynb` | 데이터 생성 규칙 및 schema 확인 |
| `02_solve_with_gurobi.ipynb` | Original CFLP, Benders, primal–dual 및 LB/UB 확인 |
| `03_build_and_validate_qubo.ipynb` | QUBO 구조·계수·encoding 및 가능한 범위의 exact validation |
| `04_run_sa.ipynb` | SA 실행 및 MP-feasible sample 평가 |
| `05_1_run_qa_pegasus.ipynb` | Pegasus controlled embedding / QA |
| `05_2_run_qa_zephyr.ipynb` | Zephyr controlled embedding / QA |
| `06_compare_results.ipynb` | Solver·embedding·QUBO 결과 시각화 |
| `10_1_theta_bound_gurobi.ipynb` | Gurobi cut trajectory의 offline bound 분석 |
| `10_2_theta_bound_sa.ipynb` | SA cut trajectory의 offline bound 분석 |
| `10_3_theta_bound_pegasus.ipynb` | Pegasus QA cut trajectory의 offline bound 분석 |
| `10_4_theta_bound_zephyr.ipynb` | Zephyr QA cut trajectory의 offline bound 분석 |
| `10_5_theta_bound_compare.ipynb` | 위 네 offline 분석 결과 종합 |

`05_1` 및 `05_2`의 기본 대상 선택은 `src/cflp_bd_qa_ex2/evaluation/target_selection.py`에 공통 정의되어 있다. Topology 비교 시 동일 인스턴스 집합과 embedding budget을 사용해야 한다. Notebook은 `src/`의 구현을 호출하는 분석 인터페이스이며, 저장된 출력 화면을 최신 실행 결과로 가정해서는 안 된다.

---

## Theta-bound offline analysis

이 기능은 `src/cflp_bd_qa_ex2/encoding/bounds.py`와 `evaluation/bound_analysis.py`, 노트북 `10_1`–`10_5`에 구현되어 있다. **SA/QPU 재샘플링이나 실제 minor embedding을 수행하지 않고** 기존에 기록된 cut pool을 이용하여 $\theta$ bound와 bit 수에 따른 QUBO의 구조적 특성을 계산한다. S3 계산 및 일부 기준해 검증에는 Gurobi가 사용된다.

### Bounds (S0–S3)

$Q(\mathbf 1)$은 **모든 시설을 개설한 상태**에서의 transportation LP 최적 비용이다.

| Strategy | $\theta_L$ | $\theta_U$ |
|---|---|---|
| **S0 — Baseline** | $0$ | $\sum_j f_j+\sum_i\sum_jd_ic_{ij}$ |
| **S1 — Cost-Based** | $0$ | $\sum_i d_i\max_j c_{ij}$ |
| **S2 — Assignment Relaxation** | $\sum_i d_i\min_j c_{ij}$ | $\sum_i d_i\max_j c_{ij}$ |
| **S3 — All-Open LP** | $Q(\mathbf 1)$ | $\sum_i d_i\max_j c_{ij}$ |

### Experiment design

- **E1-A / fixed bits**: $b_\theta=10$으로 고정하고 S0–S3를 비교한다. $\theta$ bits 수는 같지만 $\delta_\theta$와 cut-slack bits가 바뀐다.
- **E1-B / fixed target resolution**: S0의 $\delta_\theta$를 기준 목표 해상도로 삼고 각 bound에서 필요한 최소 bit 수를 계산한다. $b=\left\lceil\log_2((\theta_U-\theta_L)/\delta_{\mathrm{ref}}+1)\right\rceil$.
- **E2 / bit sweep**: $b_\theta\in\{4,6,8,10,12\}$에 따른 정밀도와 QUBO 자원 trade-off를 비교한다.
- **Full trajectory & snapshots**: 원본 iteration의 활성 cut만 복원해 전 구간 변수 증가를 계산하고, 일부 대표 iteration에서는 실제 QUBO를 구성해 edge 수와 계수 범위를 검증한다.

현재 `10_1`–`10_4` 노트북의 분석 대상은 **25×50 benchmark 다섯 개** (`b81`, `b82`, `b83`, `b84`, `b91`)이다. 각각의 notebook은 `results/benders/`, `results/sa/` 또는 `results/qa/`의 **기존 iteration CSV**를 필요로 하므로, 입력 결과가 없는 신규 환경에서는 바로 실행할 수 없다.

**실행 순서**: `10_1`–`10_4`를 입력이 있는 trajectory별로 실행한 후 `10_5`를 실행한다. `10_5`는 기본적으로 네 결과가 모두 있어야 한다. `SOURCE_CONFIGURATION_HASH`, `SOURCE_GRAPH_ID`, `TRAJECTORY_HASHES` 등 노트북의 설정을 사용해 서로 다른 실행이 섞이지 않도록 한다.

> [!CAUTION]
> Gurobi, SA, Pegasus QA, Zephyr QA의 **cut trajectory는 서로 다르다**. 동일한 iteration 번호가 동일한 QUBO 상태를 의미하지 않는다. Bound 전략의 효과는 **한 trajectory 안에서** 먼저 비교해야 한다. 또한 offline 결과에서 변수가 줄었다고 해서 embedding time·feasible rate·optimality gap이 개선되었다고 해석할 수 없다.

분석 결과 경로:

```text
results/theta_bound/analysis_v2/<analysis_hash>/
├── theta_bound_gurobi.csv
├── theta_bound_sa.csv
├── theta_bound_pegasus.csv
├── theta_bound_zephyr.csv
├── theta_bound_per_iteration_<source>.csv
└── ... (snapshot, validation, figure 등)
```

---

## Configuration

`config/experiments/pilot.yaml`은 `config/base.yaml`을 `extends`로 상속한다. 실험 설정은 깊은 병합되며, 실행 기록에 `configuration_hash`가 남는다.

| Key | Baseline value | Description |
|---|---|---|
| `benders.max_iterations` | `1000` | Iteration 상한 |
| `benders.relative_gap_tolerance` | `1e-4` | Benders 종료 gap |
| `encoding.theta.bits` | `10` | $\theta$ bits |
| `encoding.capacity_slack.delta` | `1` | 정확한 정수 capacity slack |
| `encoding.cut_slack.delta_ratio_to_theta` | `0.25` | $\delta_\mathrm{cut}/\delta_\theta$ |
| `penalty.margin` | `1.1` | $U_\mathrm{obj}$에 곱하는 penalty 계수 |
| `sa.num_reads`, `sa.sweeps` | `1000`, `1000` | SA 예산 |
| `qa.num_reads`, `qa.annealing_time` | `1000`, `20` | QA reads, anneal time (µs) |
| `embedding.timeout`, `embedding.tries` | `300`, `5` | minorminer 탐색 예산 (seconds, tries) |
| `embedding.seeds` | `2024`–`2028` | 임베딩 반복 seed |
| `output.formats` | `csv`, `parquet` | 저장 형식 |

QA solver 선택은 `qa.solvers.pegasus`와 `qa.solvers.zephyr`에 정의된다. 실제 solver id와 working graph는 계정/서비스 환경에 따라 달라질 수 있으므로 실행 시 preflight로 검증한다.

---

## Repository structure

```text
cflp-bd-qa-ex2/
├── config/
│   ├── base.yaml
│   └── experiments/pilot.yaml
├── data/
│   ├── raw/                        # Generated instances
│   └── benchmark/                  # OR-Library input + JSON
├── notebooks/
│   ├── 01_... ~ 06_...             # Baseline & QA analysis
│   └── 10_1_... ~ 10_5_...         # Theta-bound offline analysis
├── scripts/
│   ├── generate_data.py
│   ├── convert_benchmark.py
│   ├── run_gurobi.py
│   ├── run_sa.py
│   ├── run_embedding.py
│   ├── run_qa.py
│   ├── run_pilot.py
│   └── diagnose_cuts.py
├── src/cflp_bd_qa_ex2/
│   ├── benders/                     # Loop, optimality cuts, certificates
│   ├── config/                      # YAML load / schema checks
│   ├── data/                        # Data generation, benchmark, I/O
│   ├── embedding/                   # Topology, minorminer, validation
│   ├── encoding/                    # Theta/slack, S0–S3 bounds
│   ├── evaluation/                  # Metrics, figures, offline analysis
│   ├── models/                      # CFLP, MP, SP
│   ├── qubo/                        # Builder, decoder, exact checks
│   └── solvers/                     # Gurobi, SA, QA, QA master
├── tests/                           # Unit/integration tests
├── results/                         # Generated locally; gitignored
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Outputs & reproducibility

주요 결과는 기본적으로 **CSV와 Parquet** 두 형식으로 저장된다.

| Path (CSV example) | Meaning |
|---|---|
| `results/ground_truth/ground_truth.csv` | Original CFLP solve 결과 |
| `results/benders/benders_iterations.csv` | Gurobi-controlled iteration별 상태 |
| `results/benders/benders_runs.csv` | Gurobi-controlled 종료 요약 |
| `results/benders/cut_trajectory.csv` | 활성화 여부를 포함한 cut/dual 기록 |
| `results/sa/sa_iterations.csv`, `sa_runs.csv` | SA end-to-end 결과 |
| `results/qubo/qubo_metrics.csv` | QUBO 변수·edge·계수 정보 |
| `results/embedding/embedding_trials.csv` | Controlled embedding 성공/실패 trial |
| `results/embedding/mappings/` | Controlled embedding mapping JSON |
| `results/embedding/qa_embedding_trials_<topology>.csv` | QA 경로의 embedding trial |
| `results/embedding/qa_mappings/` | QA에서 사용한 mapping JSON |
| `results/qa/qa_iterations_<topology>.csv` | QA iteration별 기록 |
| `results/qa/qa_runs_<topology>.csv` | QA 종료 요약 |
| `results/qa/qa_per_embedding_<topology>.csv` | 성공한 embedding seed 및 QA 반복별 평가 |
| `results/qa/qa_preflight_<topology>.csv` | QPU 선택·접근 상태 |
| `results/theta_bound/analysis_v2/` | Bound별 offline 구조 분석 |

`<topology>`는 `pegasus` 또는 `zephyr`이다. QA 결과는 topology별 파일로 분리되며, 종료 전 중단에 대비해 **인스턴스 완료마다 checkpoint**를 저장한다.

재현성과 결과 해석에 필요한 사항:

- **세 trajectory 구분**: `gurobi_controlled`, `sa_end_to_end`, `qa_end_to_end`는 SP에 전달한 후보 $y$와 cut pool이 다를 수 있다.
- **Provenance**: `configuration_hash`, Git commit/dirty 정보, 버전 및 실행 메타데이터가 결과에 기록된다. Offline 분석은 별도로 `analysis_hash`, `trajectory_hash`를 사용한다.
- **Seed**: 생성 인스턴스 seed, SA seed, embedding seed를 구별한다. 실제 QPU sampling에는 재현 가능한 `qa_seed`를 가정하지 않는다.
- **Resource metrics**: `logical_variables`와 `physical_qubits`, embedding 성공률과 raw MP-feasible 비율, QPU 시간과 embedding 시간을 각각 구별한다. QPU timing 열의 `_us`는 **마이크로초**다.
- **Ignored outputs**: `results/`는 `.gitignore`에 포함되어 Git이 추적하지 않는다. 연구 결과와 설정 snapshot은 **별도 보관**해야 한다.

진단용 cut signature 분석은 아래처럼 실행한다.

```bash
python scripts/diagnose_cuts.py
```

---

## Validation & limitations

### Interpreting termination statuses

| Status | Interpretation |
|---|---|
| `OPTIMAL` | 지정된 Benders gap tolerance 도달 |
| `MAX_ITERATIONS` | 설정된 반복 횟수 상한 도달 |
| `STALLED_DUPLICATE_CUT` | 반복된 후보/cut과 gap 정체로 종료 |
| `NO_FEASIBLE_SAMPLE` | 해당 MP에서 feasible SA/QA sample 미발견 |
| `EMBEDDING_FAILED` | **설정된 탐색 예산 내** embedding 미발견 (불가능 증명 아님) |
| `EMBEDDING_INVALID` | embedding 검증 실패 |
| `NO_QPU_ACCESS` | QPU 접근 불가 |
| `QPU_QUOTA_EXHAUSTED` | QPU 할당 시간 소진 |
| `GUROBI_SIZE_LIMIT` | Gurobi 제한 라이선스 크기 초과 |
| `GUROBI_LICENSE_UNAVAILABLE` | Gurobi 라이선스 접근 불가 |
| `THETA_RANGE_INVALID`, `CUT_SLACK_RANGE_INVALID` | encoding 구간 유효성 문제 |
| `DUAL_NOT_MINIMAL` | Dual minimality guard 위반에 따른 진단/중단 가능 상태 |

`dual_minimality_status`, `dual_guard_action`, `termination_status`는 서로 다른 필드다. Dual guard는 반환된 $u$를 고정했을 때 $v$의 불필요한 팽창을 감지하며, 자동으로 dual을 투영해 cut을 교체하지 않는다. Gurobi-controlled trajectory에서는 설정에 따라 해당 실행을 중단할 수 있고, SA/QA는 수치적 유효성이 확인되는 경우 raw dual로 계속 진행할 수 있다.

### Scope of the current implementation

1. **Independent embedding baseline**: 매 iteration 새 embedding을 탐색한다. `initial_chains` warm start, fixed/incremental embedding 및 사전 계산한 clique embedding의 성능 비교는 아직 포함하지 않는다.
2. **Single-cut, accumulate-all**: 현재 활성 cut은 모두 유지하며 cut selection/retirement 알고리즘은 포함하지 않는다.
3. **No solution correction / reverse annealing**: Sample을 디코딩하고 MP feasibility를 검사하지만 시설 선택을 임의로 repair하지 않으며, QA는 forward-annealing 기반이다.
4. **Limited exact verification**: Exhaustive enumeration과 Gurobi MIQP는 변수 수·계산시간·라이선스 한계에 따라 건너뛰거나 종료될 수 있다. `TIME_LIMIT`에서 얻은 incumbent를 검증된 QUBO ground state로 표시하지 않는다.
5. **No automatic performance claim from offline bounds**: Theta bound 분석은 QUBO 구조·encoding 해상도의 변화를 연구한다. 실제 성공률·solution quality·embedding speed는 별도의 online experiment로 검증해야 한다.
6. **Hardware variability**: 실제 QPU의 topology, working graph, noise, 이용 가능 시간에 따라 결과가 달라질 수 있다. Minor embedding의 실패와 physical-qubit 수는 해당 문제의 수학적 한계나 최솟값을 의미하지 않는다.

---

**Code status:** Research prototype (`0.1.0`). 분석과 실행 방법은 이 저장소의 현재 `src/`, `scripts/`, `config/`, `notebooks/` 구현을 기준으로 기술하였다.
