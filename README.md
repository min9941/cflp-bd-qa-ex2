# cflp-bd-qa-ex2

Multiple-Source Capacitated Facility Location Problem(MS-CFLP)을 Benders decomposition으로
분해하고, Master Problem을 QUBO로 변환해 Simulated Annealing(SA)과 D-Wave Quantum
Annealing(QA)으로 푸는 파일럿 연구 코드이다.

---

## 1. 연구 목적

Benders iteration이 진행될 때 Master Problem QUBO와 그 embedding이 **어떻게 성장하는지**를
관측하는 것이 목적이다. 구체적으로 다음을 본다.

1. iteration 증가에 따른 logical variables / logical edges 증가
2. QUBO density와 coefficient range 변화
3. Pegasus와 Zephyr에서의 embedding 성공률 차이
4. physical qubits, chain length, embedding overhead, embedding time 증가
5. Gurobi / SA / QA의 MP feasibility와 solution quality 차이
6. 후속 연구에서 검토할 embedding 전략(초기 고정 / 매 iteration 재임베딩 / incremental)

이번 파일럿에서는 **embedding 전략이나 penalty를 사후 조정하지 않는다.**
먼저 매 iteration마다 전체 QUBO를 독립적으로 새로 임베딩하여 관측 데이터를 모은다.

우선순위는 `정확성 > 명확성 > 재현성 > 확장성 > 성능` 이다.

---

## 2. MS-CFLP 정의

시설은 `j`, 고객은 `i`로 표기한다.

```
min   sum_j f_j y_j + sum_i sum_j d_i c_ij x_ij
s.t.  sum_j x_ij = 1                      for all i
      sum_i d_i x_ij <= s_j y_j           for all j
      y_j in {0,1},  x_ij >= 0
```

`y_j`는 binary, `x_ij`는 continuous이다. 한 고객의 수요가 여러 시설로 나뉠 수 있으므로
multiple-source(MS)이다.

### `x_ij <= y_j` 를 넣지 않은 이유

baseline에 `x_ij <= y_j`를 포함하지 않는다. 근거는 두 가지이다.

1. `d_i > 0`이고 capacity constraint가 있으므로 `y_j = 0`이면
   `sum_i d_i x_ij <= 0` 에서 자동으로 `x_ij = 0`이 된다.
2. 현재 dual formulation에 해당 제약의 dual 변수가 없다.
   제약을 추가하면 dual과 cut 형태가 모두 달라진다.

테스트 `tests/test_original_model.py::test_closed_facility_receives_no_flow`가
실제로 닫힌 시설에 flow가 가지 않음을 확인한다.

---

## 3. Benders decomposition

`y`를 고정하면 남는 문제는 transportation problem이므로, `y`를 master에 두고
`x`를 subproblem으로 분리한다. Subproblem의 dual을 이용해 optimality cut을 만들고
master에 누적한다.

Lower bound는 master에서, upper bound는 subproblem을 푼 실제 `y`에서 나온다.

---

## 4. MP / SP / dual formulation

### Master Problem (MP)

```
min   sum_j f_j y_j + theta
s.t.  sum_j s_j y_j >= D
      theta + sum_j s_j v_j^k y_j >= sum_i u_i^k     for k = 1..K
      y_j in {0,1},  theta >= 0
```

`D = sum_i d_i` 는 총수요이다.

### Primal Subproblem (주어진 ybar에 대해)

```
min   sum_i sum_j d_i c_ij x_ij
s.t.  sum_j x_ij = 1                 for all i     -> dual u_i (free)
      sum_i d_i x_ij <= s_j ybar_j   for all j     -> dual v_j (>= 0)
      x_ij >= 0
```

### Dual Subproblem

```
max   sum_i u_i - sum_j s_j ybar_j v_j
s.t.  u_i - d_i v_j <= d_i c_ij
      v_j >= 0,  u_i free
```

### Optimality cut

```
theta >= sum_i u_i* - sum_j s_j v_j* y_j
```

동치 형태 `theta + sum_j s_j v_j* y_j >= sum_i u_i*` 를 코드에서 사용한다.

### Feasibility cut을 만들지 않는 이유

모든 customer-facility arc가 존재하고 `x`가 continuous이므로, aggregate capacity
constraint `sum_j s_j y_j >= D` 가 만족되면 SP는 이론적으로 항상 feasible하다.
SP가 `INFEASIBLE`로 나오면 정상적인 feasibility cut을 임의로 만들지 않고
formulation 또는 수치 문제로 기록하고 실행을 중단한다.

### Dual 비유일성과 minimality guard

닫힌 시설(`ybar_j = 0`)의 `v_j`는 dual objective에서 계수가 0이고, dual constraint는
`v_j`가 커질수록 느슨해진다. 따라서 **optimal face 위에서 위로 무한히 커질 수 있다.**

이것이 문제가 되는 이유는 정확성이 아니라 재현성이다. 팽창된 `v`로 만든 cut도
weak duality에 의해 여전히 valid하지만, `S_k^max` 가 달라져 cut slack bit 수와
QUBO 크기, embedding 결과가 solver의 dual 선택에 좌우된다.

본 프로젝트는 이를 **탐지만 하고 자동 수정하지 않는다.**

```
v_min_j = max(0, max_i (u_i / d_i - c_ij))
위반 판정: |v_j - v_min_j| > atol + rtol * max(|v_j|, |v_min_j|)
```

- guard 통과: raw Gurobi dual을 그대로 사용
- guard 실패: 자동 projection하지 않음. raw / projected dual을 모두 저장
- `gurobi_controlled`: 해당 instance/seed trajectory만 중단 (`stop_run`)
- `sa_end_to_end`, `qa_end_to_end`: raw dual로 계속 진행 (`continue_with_raw_dual`)

계속 진행은 다음이 **모두** 통과할 때만 허용한다.

1. SP status가 `OPTIMAL`
2. dual constraint가 feasible
3. primal-dual gap이 허용 오차 이내

**주의 1.** `v_min`은 "minimum-norm dual"이 아니다. 반환된 `u`를 고정했을 때의
componentwise 최소성만 보장한다. `u` 자체도 비유일할 수 있으므로 이 guard는
닫힌 시설 `v_j`의 무의미한 팽창만 탐지하며 dual 전체를 canonical하게 만들지 않는다.

**주의 2.** `dual_minimality_status` 와 `dual_guard_action` 은 **진단 field**이며
`termination_status` 를 절대 덮어쓰지 않는다. 세 값은 결과에 별도 열로 저장된다.

---

## 5. Gurobi certificate의 역할

세 모델을 명확히 구분한다.

| 계층 | 정의 | 용도 |
|---|---|---|
| `continuous_mp` | `y in {0,1}`, `theta in R` | **유효한 Benders lower bound**와 종료 판정 |
| `encoded_mp` | theta만 binary encoding, 원래 부등식 유지 | 순수한 discretization 영향 측정 |
| `penalized_qubo` | slack + constraint penalty 포함 | 실제로 SA/QA가 푸는 문제 |

SA나 QA가 MP 후보를 만들더라도, **동일 iteration의 continuous MP를 Gurobi로 별도
해결**하여 certified lower bound를 얻는다. QA/SA 목적값은 절대 lower bound로 쓰지 않는다.

Original CFLP를 Gurobi로 직접 푼 결과는 `OPTIMAL`일 때만 ground truth(`objective_opt`)로
취급한다. Time limit으로 얻은 incumbent는 ground truth라고 부르지 않는다.

---

## 6. QUBO 변환

```
F_QUBO = F_obj
       + penalty_alpha_capacity * r_capacity^2
       + sum_k penalty_alpha_cut_k * r_k^2
```

```
F_obj      = sum_j f_j y_j + theta(t)
r_capacity = sum_j s_j y_j - D - q_cap
r_k        = theta(t) + sum_j s_j v_j^k y_j - sum_i u_i^k - q_k
```

모든 변수가 binary이므로 `x^2 = x`를 이용해 제곱을 전개한다.
`src/cflp_bd_qa_ex2/qubo/builder.py`의 `AffineForm.square_into`가 이 전개를 담당하며,
`tests/test_qubo.py`가 dimod BQM energy와 일치함을 확인한다.

---

## 7. Pure-exponential encoding

```
value = offset + delta * sum_{p=0}^{b-1} 2^p z_p
```

마지막 bit를 줄이는 truncated encoding은 **사용하지 않는다.**

theta의 경우:

```
theta_L = 0
theta_U = U_obj = sum_j f_j + sum_i sum_j c_ij d_i
delta_theta = (theta_U - theta_L) / (2^b_theta - 1)
```

`theta_U = U_obj`는 매우 느슨한 상한이다. 각 고객이 실제로는 시설 하나만 이용하는데도
모든 `j`에 대해 `c_ij d_i`를 합산하기 때문이다. 이 느슨함이 뒤에 나오는
cut slack bit 수에 직접 영향을 준다.

---

## 8. Capacity slack과 cut slack의 차이

이 둘은 성격이 근본적으로 다르다.

| | capacity slack | cut slack |
|---|---|---|
| equality | `sum_j s_j y_j - D - q_cap = 0` | `theta + sum_j s_j v_j^k y_j - sum_i u_i^k - q_k = 0` |
| 계수 | `s_j`, `D` 모두 **정수** | dual `v_j`, `u_i` 는 **실수** |
| delta | `1` | `r_cut * delta_theta` (기본 `0.25 * delta_theta`) |
| 표현 | `[0, S_cap_max]` 정수를 **정확히** 표현 | 격자로 정확히 표현되지 않음 |
| bit 수 | `ceil(log2(S_cap_max + 1))` | `ceil(log2(S_k_max / delta_cut + 1))`, 자동 계산 |

```
S_cap_max = sum_j s_j - D
S_k_max   = max(0, theta_U + sum_j s_j v_j^k - sum_i u_i^k)
```

`S_k_max`는 residual을 `theta = theta_U`, `y = 1`에서 평가한 값이다
(`v_j >= 0` 이므로 이것이 최대값이다).

규칙:

- bit 수를 임의로 cap하지 않는다
- truncated last bit를 쓰지 않는다
- dual coefficient를 격자에 맞춰 반올림하지 않는다
- bit 수가 매우 크면 경고만 출력한다
- `S_k_max = 0` 이면 slack bit를 만들지 않는다
- `-1e-6 <= S_k_max < 0` 은 수치 오차로 0 처리하고 표시를 남긴다
- `S_k_max < -1e-6` 이면 `CUT_SLACK_RANGE_INVALID`로 중단한다

**dual coefficient가 실수이므로 MP-feasible sample이 penalty residual 0을 갖는다고
가정하면 안 된다.** feasibility 판정은 slack equality가 아니라 원래 MP 부등식으로 한다.

### cut slack bit 수에 대한 관찰 (empirical regularity)

`B_k = 1 - (sum_i u_i^k - sum_j s_j v_j^k) / theta_U` 로 두면

```
b_k = ceil(log2( ((2^b_theta - 1) / r_cut) * B_k + 1 ))
```

기본 설정(`b_theta = 10`, `r_cut = 0.25`)에서 `(2^10 - 1)/0.25 = 4092` 이고,
`b_k = 12` 가 되는 정확한 구간은

```
2047/4092 < B_k <= 4095/4092     즉  0.500244 < B_k <= 1.000733
```

이다. 실험한 trajectory에서는 `y = 1` 에서의 cut 값이 `U_obj` 에 비해 작아
관측된 `B_k` 가 모두 이 구간 안에 들어왔고, 따라서 모든 cut slack이 12 bit였다.

**이것은 느슨한 `theta_U` 에서 기대되는 경험적 경향이지 모든 인스턴스에 대한 이론적
보장이 아니다.** 특히 닫힌 시설의 `v_j` 가 팽창하면 `B_k > 1.000733` 이 되어
`b_k > 12` 가 될 수 있다. 그래서 dual minimality guard가 필요하다.

QUBO 변수 수의 **정의**는 다음이다.

```
n_QUBO = n_f + b_theta + b_cap + sum_{k=1}^{K} b_k^cut
```

`n_f + b_theta + b_cap + 12K` 는 관측된 empirical model일 뿐이며 정의가 아니다.

---

## 9. Constraint `penalty_alpha`

constraint penalty coefficient는 코드와 문서에서 항상 `penalty_alpha`라고 부른다.
일반 이름 `alpha`만 단독으로 쓰지 않는다.

```
penalty_alpha_capacity = margin * U_obj
penalty_alpha_cut_k    = margin * U_obj     for all k     (margin = 1.1)
```

각 constraint에 개별 값을 전달하되 파일럿에서는 모두 같다.

**구현하지 않는 것:** constraint별 penalty tuning, adaptive penalty update, penalty
sweep, Ocean의 `10 x max objective bias` 자동 규칙, constraint normalization.

작은 QUBO에서는 ground state가 MP-feasible한지 확인해 penalty 충분성을 검증한다.
부족하면 자동으로 값을 바꾸지 않고 `PENALTY_INSUFFICIENT`로 기록한다.

---

## 10. Chain strength와 constraint penalty의 차이

**둘은 전혀 다른 개념이다. 혼동하지 않는다.**

| | `penalty_alpha` | chain strength |
|---|---|---|
| 대상 | 원래 문제의 제약 위반 | 같은 logical variable을 나타내는 physical qubit 묶음 |
| 결정 시점 | QUBO를 만들 때 | **embedding 성공 후** |
| 계산 | `margin * U_obj` | `uniform_torque_compensation(bqm, embedding, prefactor=1.414)` |
| 기록 field | `penalty_alpha_capacity` 등 | `chain_strength`, `chain_strength_prefactor` |

QPU 제출 시 `auto_scale = true`로 설정한다. 다만 **uniform scaling이 coefficient
dynamic range를 개선한다고 해석하지 않는다.** 균일 스케일링은 최대/최소 비율을
바꾸지 않는다.

logical QUBO, Ising `h/J`, embedded physical problem의 coefficient 범위를 서로 섞지 않는다.
`qubo/diagnostics.py`는 **logical QUBO만** 다룬다. coefficient range는 차이가 아니라 비율이다.

```
R_Q = max_{Q_ab != 0} |Q_ab| / min_{Q_ab != 0} |Q_ab|
```

---

## 11. SA

`dwave-neal`의 `SimulatedAnnealingSampler`를 사용한다.

```yaml
sa:
  num_reads: 1000
  sweeps: 1000
  seed: 2024
```

SA seed는 configurable하며 결과에 기록된다.
Sample은 energy 오름차순으로 검사하고 현재 MP의 feasibility를 만족하는 첫 sample을 고른다.

---

## 12. QA

- **이상적 target graph에서는 QA sampling을 하지 않는다.** 실제 working graph에서만 한다.
- 성공한 embedding seed를 **모두** QA run으로 연결한다.
  가장 짧은 chain이나 가장 적은 physical qubit의 embedding만 고르지 않는다.
- 실패한 embedding 행도 결과에 보존한다.
- QPU는 재현 가능한 `qa_seed`를 제공하지 않으므로 해당 field를 쓰지 않는다.
  대신 `embedding_seed`, `qa_repeat_id`, `qpu_problem_id`, `solver_id`, `graph_id`,
  `timestamp`를 기록한다.
- Chain-break 처리는 `majority_vote`로 고정하며 다른 방법과 비교하지 않는다.

---

## 13. Pegasus와 Zephyr

| | Pegasus | Zephyr |
|---|---|---|
| 세대 | Advantage | Advantage2 |
| 이상적 크기 | P16 | Z12 |
| 노드 수 (이상적) | 5640 | 4800 |
| 엣지 수 (이상적) | 40484 | 45864 |
| 큐빗당 연결도 | 15 | 20 |

Zephyr는 노드 수가 더 적지만 연결도가 높아 조밀한 source graph를 더 짧은 chain으로
임베딩할 수 있다. 본 파일럿은 두 topology에서 같은 QUBO snapshot을 임베딩해
이 차이를 관측한다.

Zephyr solver 선택 규칙은 다음 순서를 따른다.

1. `solver_name`이 있으면 해당 solver 사용
2. 실제 topology/shape가 요구사항과 다르면 `TOPOLOGY_MISMATCH`로 중단
3. `solver_name`이 없으면 접근 가능한 Zephyr QPU 목록 조회
4. 정확히 하나이면 자동 선택
5. 0개이면 `NO_QPU_ACCESS`
6. 둘 이상이면 임의 선택하지 않음
7. solver 목록을 남기고 `AMBIGUOUS_QPU_SELECTION`

---

## 14. Ideal topology와 working graph의 차이

**이상적 target graph**는 결함이 없는 완전한 P16 / Z12이다.
**실제 working graph**는 특정 칩의 결함(죽은 큐빗, 죽은 coupler)이 반영된 그래프이다.

두 결과를 절대 섞지 않는다. 결과에는 `target_mode` (`ideal` / `actual`) 열로 구분되며,
figure에서도 분리해 그린다.

실제 working graph에서는 다음을 저장한다.

```
solver_id, chip_id, topology, topology_shape, graph_id,
target_nodes, target_edges, target_graph_fingerprint, solver_snapshot_timestamp
```

`target_graph_fingerprint`는 노드/엣지 집합 전체의 해시이므로, 같은 solver 이름이라도
결함이 달라지면 지문이 달라진다.

---

## 15. Embedding study

두 종류의 iteration 연구를 **합치지 않는다.**

### Structural controlled study

Gurobi-Benders가 만든 **동일한 cut trajectory**를 저장하고, 모든 iteration snapshot을
네 target에 임베딩한다: ideal P16, ideal Z12, actual Advantage, actual Zephyr.
각 snapshot에 embedding seed 5개를 모두 적용한다.
이 설계로 topology 차이와 search randomness를 분리해 볼 수 있다.

### End-to-end solver study

SA/QA가 만든 실제 candidate `y`로 SP를 풀고 cut을 추가한다.
solver별로 cut trajectory가 달라지므로 structural study 결과와 합치지 않는다.

결과의 `trajectory_source` 열이 `gurobi_controlled` / `sa_end_to_end` / `qa_end_to_end`
중 하나로 항상 채워진다.

### Embedding 성공 판정

성공 여부를 단순히 non-empty mapping으로 판단하지 않는다. 다음을 모두 확인한다.

- 모든 logical variable에 non-empty chain 존재
- chain 간 physical qubit 중복 없음
- 모든 physical qubit가 target graph에 존재
- 각 chain 내부가 연결됨
- 모든 logical edge를 실현하는 physical coupler 존재

Source graph는 BQM의 모든 변수를 포함하는 NetworkX graph로 만든다.
**quadratic edge가 없는 isolated logical variable도 누락하지 않는다.**

`minorminer`의 실패는 embeddability의 수학적 부재를 **증명하지 않는다.**
결과 설명에서는 반드시 "주어진 search budget에서 embedding을 찾지 못함"이라고 표현한다.
또한 관측된 한 embedding의 physical qubit 수는 해당 문제의 하한이 아니다.

파일럿에서는 `fixed_chains`, `initial_chains`, `suspend_chains`, incremental embedding,
이전 iteration embedding 재사용을 **사용하지 않는다.**

---

## 16. 프로젝트 구조

```
cflp-bd-qa-ex2/
├── README.md, requirements.txt, pyproject.toml
├── config/
│   ├── base.yaml                  # 공통 기본값
│   └── experiments/pilot.yaml     # 파일럿 인스턴스 정의
├── data/raw/                      # 생성된 인스턴스 JSON
├── results/                       # ground_truth, benders, qubo, embedding, qa, sa, figures
├── src/cflp_bd_qa_ex2/
│   ├── status.py                  # Status / DualMinimalityStatus / DualGuardAction
│   ├── logging_utils.py
│   ├── config/    {loader, schema}
│   ├── data/      {generator, io}
│   ├── models/    {cflp, master, subproblem}
│   ├── benders/   {solver, cuts, certificate, trace}
│   ├── encoding/  {pure_exponential, theta, slack}
│   ├── qubo/      {builder, penalty, diagnostics, decode, exact}
│   ├── solvers/   {gurobi, sa, qa}
│   ├── embedding/ {targets, embedder, validation, storage}
│   └── evaluation/{metrics, results, plots}
├── notebooks/  01~06
├── scripts/    generate_data, run_gurobi, run_sa, run_embedding, run_qa, run_pilot
└── tests/      10개 테스트 모듈
```

---

## 17. 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 18. Gurobi 설정

### 라이선스 크기 한계

pip로 설치되는 size-limited 라이선스에는 두 가지 한계가 있다.

| 모델 종류 | 한계 |
|---|---|
| 선형 모델 | 변수 2,000개 / 선형 제약 2,000개 |
| **이차항이 있는 모델(QP/MIQP 등)** | **변수 200개** |

본 프로젝트에서 이것이 실제로 걸리는 지점은 다음과 같다.

| 인스턴스 | Original CFLP 변수 | SP 변수 | 제한 라이선스에서 |
|---|---|---|---|
| 4x12 | 52 | 48 | 전부 가능 |
| 8x25 | 208 | 200 | 전부 가능 |
| 16x50 | 816 | 800 | 전부 가능 |
| 25x50 | 1275 | 1250 | 전부 가능 |

파일럿의 가장 큰 크기는 **25x50** 이다. SP 의 `x_ij` 가 1250개로 제한 라이선스의
선형 2000 변수 한계 안에 들어오므로, **모든 파일럿 인스턴스를 제한 라이선스에서도
실행할 수 있다.**

이전 명세의 `32x100` 은 SP 의 `x_ij` 가 3200개라 제한 라이선스에서 SP 부터 실패했다.
Original CFLP 만의 문제가 아니었다. 참고로 그 경우 다음처럼 나뉜다.

| 항목 | SP 가 한계를 넘는 크기에서 |
|---|---|
| 데이터 생성 | 가능 |
| cut-free initial MP의 QUBO 생성 / SA / ideal embedding | 가능 |
| Original CFLP ground truth | `GUROBI_SIZE_LIMIT` |
| Gurobi-controlled cut trajectory | 첫 SP에서 `GUROBI_SIZE_LIMIT` |
| SA/QA end-to-end Benders | 첫 SP에서 `GUROBI_SIZE_LIMIT` |

full license 환경에서는 전부 실행 가능하다.

**라이선스 제한을 피하려고 모델을 축소하거나 constraint를 제거하지 않는다.**

### Preflight

정적으로 "이 환경에는 라이선스가 없다"고 단정하지 않는다.
실행 시점에 실제로 확인한다.

```python
from cflp_bd_qa_ex2.solvers.gurobi import check_gurobi_available, probe_size_limits
print(check_gurobi_available())   # (available, status, message)
print(probe_size_limits())        # {'linear_limit', 'quadratic_limit', 'restricted'}
```

`scripts/run_gurobi.py`는 시작할 때 이 preflight를 자동으로 수행하고 결과를 출력한다.

---

## 19. D-Wave 접근 설정

```bash
dwave config create     # 대화형으로 token 입력
dwave ping              # 접근 확인
dwave config ls         # 설정 파일 위치 확인
```

### API token / 프로젝트 전환

세 가지 방법이 있으며, 우선순위는 **환경변수 > 프로필 > 기본 프로필** 이다.

1. **환경변수 (가장 간단)**

   ```bash
   # Linux / macOS
   export DWAVE_API_TOKEN=DEV-...
   # Windows PowerShell
   $env:DWAVE_API_TOKEN = "DEV-..."
   ```

   Jupyter 를 쓰는 경우 **커널을 띄우기 전에** 설정해야 한다.
   이미 켜진 커널에서 바꾸려면 노트북 맨 위에서:

   ```python
   import os
   os.environ["DWAVE_API_TOKEN"] = "DEV-..."   # 노트북에 평문으로 남지 않게 주의
   ```

2. **프로필 분리 (계정/프로젝트가 여럿일 때 권장)**

   ```bash
   dwave config create --profile lab_account
   dwave config create --profile personal
   ```

   설정에서 프로필을 지정한다.

   ```yaml
   qa:
     profile: "lab_account"
   ```

   `select_actual_qpu()`, `list_accessible_qpus()`, `run_qa()` 가 모두 이 값을 쓴다.

3. **기본 프로필**: `qa.profile: null` 이면 `dwave.conf` 의 기본 프로필을 쓴다.

**token 을 설정 파일(`config/*.yaml`)에 넣지 않는다.** 설정 snapshot 이
`results/config_snapshot_*.yaml` 로 저장되므로 credential 이 결과와 함께 남는다.
그래서 `qa` 에는 `profile` 만 있고 `token` 키는 없다.

### 할당량 소진 (`QPU_QUOTA_EXHAUSTED`)

solver access time 이 소진되면 D-Wave 는 다음을 돌려준다.

```
SolverFailureError: Problem not accepted because user has
insufficient remaining solver access time in project ...
```

이는 접근 권한이 없는 것(`NO_QPU_ACCESS`)과도, 제출 실패(`SOLVER_FAILED`)와도 다르다.
재시도해도 소용없으므로 별도 상태 `QPU_QUOTA_EXHAUSTED` 로 기록하고
해당 trajectory 를 중단한다.

**주의**: D-Wave 의 `sample()` 은 Future 를 즉시 돌려주고 실제 오류는
`sampleset.resolve()` 시점에 발생한다. 따라서 `run_qa()` 는 resolve 를
예외 처리 블록 **안에서** 수행한다. 그렇지 않으면 할당량 소진이
Benders 루프 전체를 중단시킨다.

토큰은 결과나 로그에 절대 출력되지 않는다 (`logging_utils.py`가 마스킹한다).

**QPU 접근 실패 시 전체 프로젝트가 crash하지 않는다.** `NO_QPU_ACCESS`로 기록하고
ideal embedding study는 계속 진행한다.

네트워크가 차단된 환경(컨테이너 등)에서는 token 유무와 무관하게 접근이 불가능하다.
이 경우에도 `run_qa.py`는 정상 종료하며 `results/qa/qa_preflight_<topology>.csv`에 상태를 남긴다.

---

## 20. 데이터 생성

### 생성 규칙

```
좌표:        시설, 고객 각각 독립적으로 U([0,1]^2)
demand:      N(mean=35, standard_deviation=5), 즉 N(35, 5^2), max(1, round(.))
capacity:    s_raw ~ U(10,160) -> 총합이 ceil(1.5 D) 가 되도록 비례 rescale 후 정수화
fixed cost:  f_j = U(0,90) + U(100,110) * sqrt(s_j)
transport:   c_ij = 10 * dist_ij
```

`demand`의 `5`는 **표준편차**이다(분산이 아니다). NumPy의 `Generator.normal(loc, scale)`의
두 번째 인자가 표준편차이므로 `rng.normal(35.0, 5.0)`이 된다.

capacity 정수화 규칙:

1. rescale 후 floor
2. 부족한 unit을 fractional remainder가 큰 시설부터 배분
3. remainder가 같으면 facility index가 작은 시설부터 배분
4. 최종적으로 `sum_j s_j == ceil(1.5 D)` 를 assert

### 생성법의 출처와 차이

capacity `U(10,160)`, `f_j = U(0,90) + U(100,110) sqrt(s_j)`, `c_ij = 10 dist`,
총용량비 `1.5` 는 Cornuejols 계열 생성법을 그대로 따른다.
다만 원래 생성법의 demand가 후속 문헌에서 주로 `U[5,35]` 로 설명되는 반면,
본 명세는 `N(35, 5^2)` 를 쓴다.

따라서 본 생성법은 **"Cornuejols 계열의 위치/비용/capacity 구조를 따르되
demand 분포를 변경한 변형"** 이라고 기술해야 정확하다.

| | Cornuejols `U[5,35]` | 본 명세 `N(35, 5^2)` |
|---|---|---|
| 평균 | 20 | 35 |
| 변동계수 | 약 0.43 | 약 0.14 |

총용량이 `1.5 D`로 재정규화되므로 **총량 비율 자체는 보존된다.**
다만 고객 간 demand 이질성이 원래보다 작다. 이는 SP의 flow 배분, 시설별 capacity
binding pattern, dual multiplier와 Benders cut 구조에 영향을 줄 수 있다.
**영향의 방향과 크기는 실험적으로 평가하며, 여기서 단정하지 않는다.**

### draw 순서

재현성을 위해 다음 순서를 고정한다. 순서를 바꾸면 같은 seed라도 다른 인스턴스가 나온다.

```
1. facility_locations  : rng.random((n_facilities, 2))
2. customer_locations   : rng.random((n_customers, 2))
3. demand_raw           : rng.normal(35.0, 5.0, n_customers)
4. capacity_raw         : rng.uniform(10, 160, n_facilities)
5. fixed_cost_base      : rng.uniform(0, 90, n_facilities)
6. fixed_cost_slope     : rng.uniform(100, 110, n_facilities)
```

`numpy.random.Generator(PCG64(seed))`를 사용한다.

### 실행

```bash
python scripts/generate_data.py --config config/experiments/pilot.yaml
```

크기는 `facilities x customers` 순서로 표기한다 (`4x12` = 시설 4, 고객 12).

---

## 21. Notebook 실행 순서

Notebook은 핵심 algorithm을 구현하지 않는다. `src/` 함수를 호출하고 결과를 해석한다.

| Notebook | 내용 |
|---|---|
| `01_generated_instances.ipynb` | 인스턴스 생성, seed/schema 검증, 분포 요약, 총용량 assert, 재실행 결정성 |
| `02_solve_with_gurobi.ipynb` | Original CFLP, Benders+Gurobi, objective/`y` 비교, primal-dual 비교, LB/UB figure |
| `03_build_and_validate_qubo.ipynb` | iteration별 QUBO, variable-role table, 계수 통계, 세 계층 비교, exhaustive validation |
| `04_run_sa.ipynb` | SA Benders, energy 순 sample 검사, MP-feasible 선택, Gurobi 기준 비교, 실패 상태 보존 |
| `05_1_run_qa_pegasus.ipynb` / `05_2_run_qa_zephyr.ipynb` | QPU 목록, topology/graph identity 검증, controlled embedding, end-to-end QA, chain strength, chain-break, ideal/actual 분리 |
| `06_compare_results.ipynb` | 요구 figure 생성 |

### 05_1 / 05_2 를 나눈 이유

한 notebook 에서 두 topology 를 연속 실행하면 앞쪽이 끝나야 뒤쪽이 시작된다.
Pegasus 20개가 수십 시간 걸리면 Zephyr 는 시작도 못 한다.

결과의 merge key 에 `graph_id` 와 `topology_label` 이 포함되므로
**따로 실행해도 같은 파일에 안전하게 합쳐지며 덮어쓰지 않는다.**

```
qa_runs           merge key: instance_id, trajectory_source, graph_id, configuration_hash
qa_iterations     merge key: ... + benders_iteration
qa_per_embedding  merge key: ... + embedding_seed, qa_repeat_id, solver_id
```

두 notebook 은 `TARGET_TOPOLOGY` 값만 다르고 나머지 코드는 같다.
대상 인스턴스 기본값도 `evaluation/target_selection.py` 한 곳에 두고 양쪽이 이를 쓴다.
한쪽만 바꾸면 topology 효과와 instance 효과가 섞이므로, 실행 시
`compare_with_other_topology()` 가 다른 topology 의 이전 결과와 대상을 비교해
불일치를 경고한다.

QA 결과는 인스턴스 하나가 끝날 때마다 `save_qa_checkpoint()` 로 즉시 저장된다.
마지막에 한꺼번에 저장하면 인터럽트나 커널 재시작 시 모든 행이 소실된다.

결과 파일에는 topology 접미사가 붙는다(`qa_runs_pegasus.csv` 등).
`scripts/run_qa.py` 도 같은 접미사를 쓰므로 접미사 없는 구형 파일은 생성되지 않는다.
구형 파일이 남아 있으면 06 에서 같은 실행이 두 번 집계되므로 삭제한다.
실제 sampler 로직은 양쪽 모두 `solvers/qa_master.py` 의
`QAMasterProblemSolver` 를 호출하므로 중복 구현이 아니다.

**주의**: topology 비교가 목적이면 `embedding.timeout`, `tries`, `seeds` 등
탐색 예산을 양쪽 동일하게 유지해야 한다. 한쪽만 줄이면 비교가 성립하지 않는다.

---

## 22. Script 실행

```bash
python scripts/generate_data.py
python scripts/run_gurobi.py --encoded-mp
python scripts/run_sa.py
python scripts/run_embedding.py                  # ideal target만
python scripts/run_embedding.py --include-actual # 실제 QPU 포함
python scripts/run_qa.py --topology pegasus
python scripts/run_pilot.py                      # 전체 순차 실행
```

각 스크립트는 `--instances 4x12_s100,8x25_s200` 형태로 범위를 좁힐 수 있다.

---

## 23. 결과 schema

CSV와 Parquet을 모두 저장한다. 기존 field의 의미를 변경하지 않는다.

| 파일 | 내용 |
|---|---|
| `results/ground_truth/ground_truth.csv` | Original CFLP 결과 |
| `results/benders/benders_iterations.csv` | `record_type = benders_iteration` |
| `results/benders/benders_runs.csv` | `record_type = benders_run` |
| `results/benders/cut_trajectory.csv` | iteration별 cut 계수와 dual |
| `results/qubo/qubo_metrics.csv` | QUBO 크기/밀도/계수 범위 |
| `results/sa/sa_iterations.csv`, `sa_runs.csv` | SA end-to-end |
| `results/embedding/embedding_trials.csv` | 모든 trial (성공/실패 모두) |
| `results/embedding/mappings/*.json` | 성공한 embedding mapping |
| `results/qa/qa_iterations_<topology>.csv`, `qa_runs_<topology>.csv` | QA end-to-end trajectory |
| `results/qa/qa_preflight_<topology>.csv` | QPU 선택 상태 (실제 status 그대로) |
| `results/qa/qa_per_embedding_<topology>.csv` | **embedding seed별** QA 평가 (아래 참조) |
| `results/embedding/qa_embedding_trials_<topology>.csv` | QA 경로가 실제로 사용한 embedding trial (실패 포함) |

QA 결과 파일에는 **topology 접미사가 붙는다** (`qa_runs_pegasus.csv` 등).
구형 무접미사 파일이 함께 있으면 같은 실행이 두 번 집계되므로,
`load_qa_results()` 가 topology 파일이 있을 때 구형 파일을 무시하고 경고한다.
| `results/embedding/qa_mappings/*.json` | QA 제출에 실제로 쓰인 embedding mapping |
| `results/embedding/actual_embedding_preflight.csv` | actual working graph 선택 상태 |
| `results/benders/cut_signature_diagnosis.csv` | duplicate cut 패턴 진단 |

### `qa_per_embedding.csv` 를 따로 두는 이유

QA 실험 설계는 **선택 A** 이다.

- 동일 QUBO 에서 성공한 **모든** embedding seed 를 독립적으로 decode·평가한다
- sample 을 하나의 pool 로 합치지 않는다 (합치면 사실상 "가장 좋은 embedding 고르기" 가 된다)
- Benders trajectory 는 ``embedding.seeds`` 순서에서 **성공한 첫 seed** 로 고정한다

따라서 seed 별 `best_energy`, `mp_feasible_sample_rate`, `select_status`,
`selected_decoded_y`, `selected_decoded_theta`, 위반량이 모두 이 파일에 남는다.
`used_for_trajectory` 열이 어느 seed 가 trajectory 를 결정했는지 표시한다.

seed 마다 완전히 독립된 end-to-end trajectory 를 돌리는 방식(선택 B)은
**본 파일럿 범위가 아니며** 별도 실험으로 분리해야 한다.

QA 경로의 계층은 다음과 같다.

```
QAMasterProblemSolver          solvers/qa_master.py   실험 오케스트레이션
        |                                             (embedding 탐색 -> seed별 실행
        |                                              -> 독립 평가 -> 후보 선택)
        +-- run_qa()           solvers/qa.py          D-Wave 호출 래퍼
                |                                     (chain strength 계산,
                |                                      chain-break 집계, 예외 분류)
                +-- DWaveSampler / FixedEmbeddingComposite    실제 QPU 제출
```

`QAMasterProblemSolver` 는 **Ocean 의 sampler 가 아니다.**
`dimod.Sampler` 를 상속하지 않고 `.sample(bqm)` 도 제공하지 않는다.
Benders 루프가 매 iteration 호출하는 MP 후보 생성기이며,
실제 QPU 제출은 한 단계 아래 `run_qa()` 가 담당한다.

이 클래스는 `solvers/qa_master.py` 한 곳에만 있고 `scripts/run_qa.py` 와
`notebooks/05_1_run_qa_pegasus.ipynb` 와 `notebooks/05_2_run_qa_zephyr.ipynb` 가
**같은 클래스**를 호출한다 (명세 24절).

### QPU timing 열

D-Wave 의 timing 은 dict 로 오지만 그대로 저장하면 CSV 에서 문자열이 되어
집계할 때마다 파싱해야 한다. 따라서 개별 열로 펼쳐 저장한다.
**단위는 원본 그대로 마이크로초이며 열 이름의 `_us` 가 이를 나타낸다.**

```
qa_qpu_access_time_us               전체 QPU 점유 시간
qa_qpu_sampling_time_us             샘플링 시간
qa_qpu_anneal_time_per_sample_us    sample 당 anneal 시간
qa_qpu_readout_time_per_sample_us   sample 당 readout 시간
qa_qpu_programming_time_us          프로그래밍 시간
qa_qpu_delay_time_per_sample_us     sample 당 지연
qa_qpu_access_overhead_time_us      접근 overhead
qa_total_post_processing_time_us    후처리 시간
qa_post_processing_overhead_time_us 후처리 overhead
```

표준 key 는 값이 없어도 열을 만들고 `None` 으로 둔다. 실행마다 열 구성이
달라지지 않게 하기 위해서다. D-Wave 가 표준 목록 밖의 숫자 timing 을 주면
같은 규칙(`qa_<key>_us`)으로 열이 추가된다.

별도로 다음 두 값이 있다.

- `qpu_runtime` : `qpu_access_time` 을 **초** 로 바꾼 요약값.
  Benders run-level 누적(`num_qpu_calls`, `qpu_runtime`)에 쓰인다.
  측정값이 없으면 `0.0` 이 아니라 `None` 이다.
- `qa_wall_runtime` : 큐 대기와 네트워크를 포함한 벽시계 시간(초).
  `qpu_access_time` 과 크게 다를 수 있으므로 섞지 않는다.

### `used_for_qa` 와 `qa_requested`

embedding trial 기록에서 두 값은 다르다.

- `qa_requested = True` : 이 trial 이 QA 실행을 위해 시도되었다
- `used_for_qa = trial.success` : 실제로 QPU 제출에 사용되었다

실패한 mapping 은 제출에 쓸 수 없으므로 `used_for_qa = False` 이다.

**MP-level 결과와 Benders-level 결과는 서로 다른 `record_type`으로 구분된다.**

### 분리해 저장하는 값들

혼동하기 쉬우므로 각각 별도 열로 저장한다.

```
energy                        QUBO energy (penalty 포함)
decoded_mp_objective          디코딩된 y, theta의 MP 목적값
true_sp_value_for_decoded_y   해당 y에서 SP를 실제로 푼 값 Q(y)
candidate_original_objective  sum_j f_j y_j + Q(y)
continuous_mp_objective       certified lower bound의 출처
encoded_mp_objective          discretization 영향 측정용
qubo_ground_state_energy      exhaustive 또는 MIQP로 구한 ground state
```

MP에는 `x`가 없으므로 `decoded_x`를 MP metric으로 쓰지 않는다.
Original CFLP objective는 반드시 SP에서 얻은 continuous `x`를 사용해 계산한다.

### Provenance

모든 실행에 다음을 저장한다.

```
project_version, git_commit, git_dirty, git_diff_hash,
python_version, package_versions, configuration_hash,
instance_seed, sa_seed, embedding_seed,
solver_id, graph_id, timestamp
```

`git_commit` 만으로는 부족하다. commit 을 기록해도 working tree 가 더러우면
그 결과는 해당 commit 의 상태에서 생성된 것이 아니다. 따라서 다음을 함께 남긴다.

- `git_dirty` : uncommitted 변경(tracked diff 또는 untracked 파일)이 있었는가
- `git_diff_hash` : 변경분의 SHA-256 앞 16자리.
  tracked diff 내용과 untracked 파일의 **경로 + 내용 해시**를 모두 포함하므로,
  untracked 파일의 내용만 바뀌어도 값이 달라진다. 깨끗하면 `None`.

`git_dirty = True` 인 결과는 재현 근거로 쓸 수 없다.

설정 파일의 완전한 snapshot도 `results/config_snapshot_*.yaml`에 저장된다.
`--max-iterations` 등으로 설정을 덮어쓰면 변경된 설정의 snapshot 이 새로 저장된다.

### `results/` 는 Git 이 추적하지 않는다

`results/` 는 `.gitignore` 에 있다. 결과를 쓸 때마다 working tree 가 더러워져
모든 결과의 `git_dirty` 가 True 가 되는 것을 막기 위해서다.

**따라서 Git 은 실험 결과를 보관하지 않는다.** 결과 보존은 별도로 해야 한다.

- 의미 있는 실행이 끝나면 `results/` 전체를 외부 저장소나 백업에 보관한다
- 보관 시 `results/config_snapshot_*.yaml` 을 함께 둔다
- 어떤 코드에서 나온 결과인지는 CSV 의 `git_commit` / `git_dirty` 로 추적한다
- 설정을 바꿔 재실행하면 `configuration_hash` 가 달라지고, merge key 에
  `configuration_hash` 가 포함되므로 이전 결과를 덮어쓰지 않는다

---

## 24. Figure 해석

`06_compare_results.ipynb`가 생성하는 figure는 다음과 같다.

| # | 파일 | 읽는 법 |
|---|---|---|
| 1 | `fig01_final_gap_by_solver` | solver별 최종 목적값 gap. ground truth가 `OPTIMAL`일 때만 계산됨 |
| 2 | `fig02_mp_feasible_rate` | iteration에 따른 MP-feasible sample 비율. 낮아지면 cut이 늘어 QUBO가 어려워진 것 |
| 3 | `fig03_bound_trajectory` | LB는 단조 증가해야 하고 UB 아래에 있어야 함 (대표 인스턴스 1개) |
| 3 | `fig03_bound_trajectory_<instance_id>` | 인스턴스별 파일. `plot_bound_trajectory(..., suffix=True)` |
| 4-5 | `fig04/05_logical_variables/edges` | iteration당 증가량. 기울기가 `b_theta + log2(1/r_cut)` 부근인지 확인 |
| 6 | `fig06_qubo_density` | cut이 늘수록 density가 어떻게 변하는지 |
| 7 | `fig07_coefficient_range` | **비율**이며 로그 축. QA 난이도의 핵심 지표 |
| 8 | `fig08_physical_qubits` | embedding 자원 증가 |
| 9 | `fig09a/b_chain_length` | mean/max chain length |
| 10 | `fig10_embedding_time` | 탐색 시간 증가 |
| 11 | `fig11_embedding_success_heatmap_micro` | **trial 가중(micro)** 성공률. iteration 이 많은 instance seed 가 더 큰 가중치를 받는다 |
| 11 | `fig11_embedding_success_heatmap_macro` | **instance 가중(macro)** 성공률. instance seed 별 성공률을 먼저 구한 뒤 동일 가중 평균 |
| 11b | `fig11b_embedding_success_by_iteration` | iteration 별 성공률. 평균으로 숨기지 않고 따로 본다 |
| 12 | `fig12_ideal_vs_actual` | 이상적 그래프와 실제 working graph 비교 |
| 13 | `fig13_chain_break_vs_feasibility` | chain-break와 MP feasibility 관계 (QA 필요) |
| 14 | `fig14_slack_bit_composition` | theta / capacity / cut slack bit 구성 변화 |

**성공률, 자원 사용량, solution quality, MP feasibility를 하나의 "우승" metric으로
합치지 않는다.** 각각 별도로 본다.

heatmap 은 세 실험 축(instance seed, Benders iteration, embedding seed)을 평균한다.
따라서 "instance seed와 embedding seed가 분리되어 있다"고 설명하면 안 된다.
micro 와 macro 를 모두 보고, iteration 축은 `fig11b` 에서 따로 확인한다.
집계 자체는 `evaluation.plots.embedding_success_table()` 로 분리되어 있어
figure 없이도 표를 직접 검증할 수 있다.

---

## 25. 재현성

같은 설정과 같은 seed는 같은 결과를 준다. 단 다음은 예외이다.

- **QPU 샘플링**: 재현 가능한 seed가 제공되지 않는다.
  대신 `qpu_problem_id`, `solver_id`, `graph_id`, `timestamp`로 추적한다.
- **`minorminer` 탐색 시간**: `embedding_time`은 머신 부하에 따라 달라진다.
  embedding 자체는 `random_seed`로 결정적이다.

재현에 필요한 것:

```
configuration_hash + instance_seed + sa_seed + embedding_seed
```

---

## 26. 실패 status 해석

**실패를 성공으로 바꾸거나 실패 행을 결과에서 삭제하지 않는다.**

| Status | 의미 | 대응 |
|---|---|---|
| `OPTIMAL` | relative gap tolerance 도달 | - |
| `MAX_ITERATIONS` | iteration 상한 도달 | `max_iterations` 조정 검토 |
| `TIME_LIMIT` | Gurobi 개별 solve 시간 초과 | ground truth로 쓰지 않음 |
| `STALLED_DUPLICATE_CUT` | 같은 cut이 반복되고 gap이 줄지 않음 | **임의 heuristic 적용 금지.** formulation 재검토 |
| `NO_FEASIBLE_SAMPLE` | 모든 sample이 MP-infeasible | repair 금지. penalty/encoding 재검토 |
| `PENALTY_INSUFFICIENT` | ground state가 MP-infeasible | 자동 변경 금지. 별도 연구 결정 |
| `EMBEDDING_FAILED` | 주어진 search budget에서 embedding 미발견 | **비임베딩성 증명 아님** |
| `EMBEDDING_INVALID` | mapping이 검증을 통과하지 못함 | 구현 오류 신호 |
| `NO_QPU_ACCESS` | QPU 접근 불가 | ideal study는 계속 진행 |
| `QPU_QUOTA_EXHAUSTED` | solver access time 소진 | 재시도 무의미. token/프로젝트 전환 후 재실행 |
| `AMBIGUOUS_QPU_SELECTION` | 조건에 맞는 QPU가 둘 이상 | `solver_name` 명시 |
| `TOPOLOGY_MISMATCH` | 실제 topology/shape가 설정과 다름 | 설정 수정 |
| `CUT_SLACK_RANGE_INVALID` | `S_k_max < -1e-6` | formulation 또는 수치 오류 |
| `THETA_RANGE_INVALID` | theta 범위가 유효하지 않음 | `theta_U` 재검토 |
| `INFEASIBLE` (SP) | SP가 infeasible | **feasibility cut 임의 생성 금지.** 중단 |
| `GUROBI_SIZE_LIMIT` | 제한 라이선스 초과 | 모델 축소 금지. full license 필요 |
| `GUROBI_LICENSE_UNAVAILABLE` | 라이선스 없음 | - |
| `EXACT_QUBO_SKIPPED_TOO_LARGE` | 정확 검증 크기 초과 | 임의 축소 금지 |
| `DUAL_NOT_MINIMAL` | dual guard 위반 | 진단 field. `termination_status`와 별개 |

`dual_minimality_status`, `dual_guard_action`, `termination_status`는 **서로 다른 열**이다.
예를 들어 다음 조합이 정상적으로 나올 수 있다.

```
dual_minimality_status = DUAL_NOT_MINIMAL
dual_guard_action      = CONTINUE_WITH_RAW_DUAL
termination_status     = OPTIMAL
```

---

## 27. 현재 파일럿의 한계

0. **exact validation 은 가장 비싼 단계이며 세 가지 한계가 겹친다.**

   | 한계 | 설정 | 의미 |
   |---|---|---|
   | exhaustive | `exhaustive_max_variables: 25` | 2^25 열거 한계. 22 변수에 약 70초 |
   | MIQP 라이선스 | `miqp_max_variables: 200` | 제한 라이선스의 이차항 변수 상한 |
   | MIQP 계산 가능성 | `miqp_tractable_max_variables: 50` | **라이선스와 다른 개념.** penalized QUBO 는 계수 dynamic range 가 10^8 수준이라 50 변수를 넘으면 time limit 까지 끌려간다 |
   | 시간 | `exact_time_limit: 60` | `gurobi.time_limit`(600) 과 분리. 초과 시 `TIME_LIMIT` |

   `exact_skip_reason` 열이 `license_limit` / `intractable` 을 구분한다.
   `TIME_LIMIT` 으로 끝난 경우 incumbent 는 `exact_incumbent_energy` 에 남지만
   **ground state 라고 부르지 않는다** (`qubo_ground_state_energy` 는 비어 있다).

1. **정확 검증(exhaustive) 범위가 매우 좁다.**
   threshold 25 기준으로 4x12의 cut-free initial MP(22 변수)만 열거 가능하다.
   cut이 하나만 추가되어도 34 변수가 되어 불가능해진다.
   8x25(27)와 16x50(36)은 초기 MP부터 이미 초과한다.
   따라서 `EXACT_QUBO_SKIPPED_TOO_LARGE`가 사실상 기본 경로이며,
   penalty 충분성 검증도 같은 범위로 제한된다.

2. **Gurobi-controlled trajectory가 짧다.**
   Benders가 대체로 2~10 iteration에서 수렴하므로 iteration별 성장 곡선의
   데이터 포인트가 적다. SA/QA end-to-end trajectory는 더 길어질 수 있다.

3. **파일럿 최대 크기는 25x50 이다.** SP 의 `x_ij` 가 1250개로 제한 라이선스
   한계 안에 들어오므로 모든 인스턴스를 제한 라이선스에서도 실행할 수 있다.
   대신 25x50 의 QUBO 는 `y` 가 25개라 cut 이 만드는 clique 이 커서
   logical edge 밀도가 16x50 보다 훨씬 높다(초기 MP 기준 0.60 대 0.52).
   변수 수가 아니라 밀도가 embedding 난이도를 좌우하므로 크기가 작다고
   임베딩이 쉽다고 단정할 수 없다.

4. **`theta_U = U_obj`가 매우 느슨하다.**
   관측된 `b_k = 12`는 이 느슨함에 크게 의존한다.
   bound 전략 비교는 이번 baseline의 범위 밖이다.

5. **dual guard는 부분적 정규화만 탐지한다.**
   닫힌 시설 `v_j`의 팽창은 탐지하지만 `u`의 비유일성은 다루지 않는다.
   완전한 canonicalization은 Benders cut trajectory 자체를 바꾸므로
   별도 실험으로 분리해야 한다.

6. **embedding 전략 비교가 없다.**
   매 iteration 독립 재임베딩만 수행한다. warm-start(`initial_chains`)나
   incremental embedding은 후속 연구 대상이다.

7. **QA 결과는 QPU 접근 가능 환경에서만 얻을 수 있다.**
   접근 불가 환경에서는 ideal embedding study와 SA만 실행된다.

8. **`minorminer` 실패는 비임베딩성을 증명하지 않으며,
   관측된 한 embedding의 physical qubit 수는 해당 문제의 하한이 아니다.**
