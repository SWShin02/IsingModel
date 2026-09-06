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
# 2차원 이징 모형(H=0)에서 자화 순서매개변수 <|m|>를 측정하기 위해 절대값 취함
if magnetization.ndim == 3:
    m_samples = np.abs(magnetization[:, :, -1])
elif magnetization.ndim == 2:
    m_samples = np.abs(magnetization)
else:
    raise ValueError(f"예상치 못한 magnetization 데이터 차원입니다: {magnetization.shape}")

# repetition 수 N (독립 표본 수)
N = m_samples.shape[1]

# 각 온도별 평균 자화: <|m|>
m_temperature = np.mean(m_samples, axis=1)

# 표본 표준편차 (불편추정량, ddof=1): \sigma^2 = \frac{1}{N-1}\sum (x_i - \bar{x})^2
sigma = np.std(m_samples, axis=1, ddof=1)

# 표준오차: standard error = \sigma / \sqrt{N}
m_err = sigma / np.sqrt(N)

#endregion
# ===================================================================
#region Plot
# ===================================================================
# Onsager 이론 임계 온도 (J=1, k_B=1): Tc = 2 / ln(1 + sqrt(2)) ~ 2.269
Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

# 에러바 플롯
ax.errorbar(
    T_array,
    m_temperature,
    yerr=m_err,
    linestyle='none',
    marker='o',
    markerfacecolor='none',
    color='C0',
    capsize=3,
    label="magnetization"
)

# 이론적 임계 온도선 표시
ax.axvline(Tc, color='gray', linestyle='--', label=f"$T_c \\approx {Tc:.3f}$")

# 축 설정
ax.set_xlabel(r"$T$")
ax.set_ylabel(r"$\langle |m| \rangle$")
ax.set_xlim(0, T_array.max())
ax.set_ylim(0, 1)

# 틱 방향 설정 (안쪽으로, major/minor 모두 적용)
ax.tick_params(axis='both', which='both', direction='in')

# 범례
ax.legend(loc='upper right')

# 출력
fig.tight_layout()
fig.savefig("magnetization.png", dpi=300, bbox_inches='tight')
print("Saved plot to 'magnetization.png'")

#endregion