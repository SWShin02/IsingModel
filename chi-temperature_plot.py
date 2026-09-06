import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"mathtext.fontset": "cm"})

# ===================================================================
#region Load HDF5 data
# ===================================================================
h5_filename = "Ising2D.h5"

if not os.path.exists(h5_filename):
    raise FileNotFoundError(
        f"'{h5_filename}' 파일을 찾을 수 없습니다. "
        "Ising2D_simulation.py 시뮬레이션이 완료된 후 실행해 주세요."
    )

with h5py.File(h5_filename, "r") as f:
    group_names = sorted(list(f.keys()))
    if not group_names:
        raise ValueError(f"'{h5_filename}' 내에 데이터 그룹이 존재하지 않습니다.")

    # 가장 최근에 시뮬레이션된 그룹 선택 (그룹명 형식: %Y%m%d%H%M)
    latest_group_name = group_names[-1]
    print(f"Loading simulation data from group: '{latest_group_name}'")
    group = f[latest_group_name]

    # temperature: shape (len(T_array),)
    # magnetization: shape (len(T_array), repetition, iterations//10)
    T_array = group["temperature"][:]
    magnetization = group["magnetization"][:]

#endregion
# ===================================================================
#region Statistical analysis
# ===================================================================
# 시간 방향(마지막 축)의 마지막 결과만 추출 -> shape: (len(T_array), repetition)
if magnetization.ndim == 3:
    m_samples = magnetization[:, :, -1]
elif magnetization.ndim == 2:
    m_samples = magnetization
else:
    raise ValueError(f"예상치 못한 magnetization 데이터 차원입니다: {magnetization.shape}")

# 시뮬레이션 격자 크기 (Ising2D_simulation.py의 N_lat와 동일해야 함)
N_lat = 16
N_points = N_lat * N_lat

# 각 온도별 <|m|>, <m^2>
m_temperature = np.mean(np.abs(m_samples), axis=1)
m_square_temperature = np.mean(m_samples**2, axis=1)

# 자화율: \chi_T = \frac{N}{T} \left( \langle m^2 \rangle - \langle |m| \rangle^2 \right)
chi_T = (N_points / T_array) * (m_square_temperature - m_temperature**2)

# repetition 수 N (독립 표본 수)
N = m_samples.shape[1]

# chi_T는 <|m|>, <m^2>의 비선형 조합이므로 잭나이프(jackknife)로 표준오차 추정
# leave-one-out 평균: shape (len(T_array), N)
sum_abs_m = np.sum(np.abs(m_samples), axis=1, keepdims=True)
sum_m_square = np.sum(m_samples**2, axis=1, keepdims=True)

m_temperature_loo = (sum_abs_m - np.abs(m_samples)) / (N - 1)
m_square_temperature_loo = (sum_m_square - m_samples**2) / (N - 1)

chi_T_loo = (N_points / T_array[:, None]) * (
    m_square_temperature_loo - m_temperature_loo**2
)

# 잭나이프 표준오차: \sigma_{jk} = \sqrt{\frac{N-1}{N} \sum (x_i - \bar{x}_{jk})^2}
chi_T_err = np.sqrt(
    (N - 1) / N * np.sum((chi_T_loo - np.mean(chi_T_loo, axis=1, keepdims=True))**2, axis=1)
)

#endregion
# ===================================================================
#region Plot
# ===================================================================
# Onsager 이론 임계 온도 (J=1, k_B=1): Tc = 2 / ln(1 + sqrt(2)) ~ 2.269
Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

ax.errorbar(
    T_array,
    chi_T,
    yerr=chi_T_err,
    linestyle='none',
    marker='o',
    markerfacecolor='none',
    color='C0',
    capsize=3,
    label="susceptibility"
)

# 이론적 임계 온도선 표시
ax.axvline(Tc, color='gray', linestyle='--', label=f"$T_c \\approx {Tc:.3f}$")

# 축 설정
ax.set_xlabel(r"$T$")
ax.set_ylabel(r"$\chi_T$")
ax.set_xlim(0, T_array.max())
ax.set_ylim(0,20)

# 틱 방향 설정 (안쪽으로, major/minor 모두 적용)
ax.tick_params(axis='both', which='both', direction='in')

# 범례
ax.legend(loc='upper right')

# 출력
fig.tight_layout()
fig.savefig("chi_T.png", dpi=300, bbox_inches='tight')
print("Saved plot to 'chi_T.png'")

#endregion
