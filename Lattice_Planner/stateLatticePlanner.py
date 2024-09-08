import math
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

# ロボットの動力学モデル
def robot_dynamics(state, t, params):
  """
  ロボットの動力学モデル
  """
  x, y, theta, v, omega = state
  a = params[0] + params[1] * t + params[2] * t**2
  alpha = params[3] + params[4] * t + params[5] * t**2
  x_dot = v * math.cos(theta)
  y_dot = v * math.sin(theta)
  theta_dot = omega
  v_dot = a
  omega_dot = alpha
  return [x_dot, y_dot, theta_dot, v_dot, omega_dot]

# Model Predictive Trajectory Generation
def model_predictive_trajectory_generation(current_state, terminal_state, params, time_horizon, dt):
  """
  Model Predictive Trajectory Generation を用いてパスを生成する

  Args:
      current_state: ロボットの現在状態 [x, y, θ, v, ω]
      terminal_state: 目標終端状態 [x, y, θ, v, ω]
      params: 入力ベクトル `u(p, t)` のパラメータベクトル `p`
      time_horizon: 時間軸
      dt: 時間刻み

  Returns:
      trajectory: パス (状態ベクトルのリスト) [[x, y, θ, v, ω], ...]
  """

  def robot_dynamics(state, t, params):
    """
    ロボットの動力学モデル
    """
    x, y, theta, v, omega = state
    a = params[0] + params[1] * t + params[2] * t**2
    alpha = params[3] + params[4] * t + params[5] * t**2
    x_dot = v * math.cos(theta)
    y_dot = v * math.sin(theta)
    theta_dot = omega
    v_dot = a
    omega_dot = alpha
    return [x_dot, y_dot, theta_dot, v_dot, omega_dot]

  def objective_function(params):
    """
    終端状態の誤差を計算する
    """
    trajectory = odeint(robot_dynamics, current_state, time_horizon, args=(params,))
    terminal_error = np.array(terminal_state) - trajectory[-1]
    return np.linalg.norm(terminal_error)

  # ヤコビアンの近似計算
  def jacobian_of_objective_function(params):
    eps = 1e-6
    jac = np.zeros_like(params)
    for i in range(len(params)):
      params_plus = params.copy()
      params_plus[i] += eps
      jac[i] = (objective_function(params_plus) - objective_function(params)) / eps
    return jac

  # パラメータ最適化 (Newton法)
  result = minimize(objective_function, params, method='Newton-CG', jac=jacobian_of_objective_function)

  # 最適化されたパラメータを用いて軌道を生成
  trajectory = odeint(robot_dynamics, current_state, time_horizon, args=(result.x,))

  return trajectory

# Uniform Polar Sampling
def uniform_polar_sampling(current_state, distance_range, angle_range, num_distances, num_angles):
  """
  Uniform Polar Sampling で終端状態をサンプリングする

  Args:
      current_state: ロボットの現在状態 [x, y, θ]
      distance_range: サンプリングする距離の範囲 [min_distance, max_distance]
      angle_range: サンプリングする角度の範囲 [min_angle, max_angle]
      num_distances: サンプリングする距離の数
      num_angles: サンプリングする角度の数

  Returns:
      terminal_states: 終端状態のリスト [[x, y, θ], ...]
  """

  terminal_states = []
  for i in range(num_distances):
    distance = distance_range[0] + (distance_range[1] - distance_range[0]) * i / (num_distances - 1)
    for j in range(num_angles):
      angle = angle_range[0] + (angle_range[1] - angle_range[0]) * j / (num_angles - 1)
      x = current_state[0] + distance * math.cos(angle + current_state[2])
      y = current_state[1] + distance * math.sin(angle + current_state[2])
      theta = current_state[2] + angle
      terminal_states.append([x, y, theta])
  return terminal_states

# 衝突チェック (簡略化)
def check_collision(trajectory, obstacles):
  """
  パスが障害物と衝突するかどうかを判定する (簡略化)

  Args:
      trajectory: パス (状態ベクトルのリスト) [[x, y, θ, v, ω], ...]
      obstacles: 障害物のリスト [[x, y, radius], ...]

  Returns:
      collision: 衝突する場合は True、衝突しない場合は False
  """
  for state in trajectory:
    x, y, _, _, _ = state
    for obstacle in obstacles:
      xo, yo, radius = obstacle
      if math.hypot(x - xo, y - yo) <= radius:
        return True
  return False

# コスト計算 (簡略化)
def calculate_cost(trajectory):
  """
  パスのコストを計算する (簡略化)

  Args:
      trajectory: パス (状態ベクトルのリスト) [[x, y, θ, v, ω], ...]

  Returns:
      cost: パスのコスト
  """
  cost = 0
  for i in range(len(trajectory) - 1):
    x1, y1, _, _, _ = trajectory[i]
    x2, y2, _, _, _ = trajectory[i + 1]
    cost += math.hypot(x2 - x1, y2 - y1)  # 移動距離をコストとする
  return cost

# パス選択
def select_best_path(trajectories, costs):
  """
  コストが最小となるパスを選択する

  Args:
      trajectories: パスのリスト [[[x, y, θ, v, ω], ...], ...]
      costs: 各パスのコストのリスト [cost, ...]

  Returns:
      best_trajectory: 最適なパス [[x, y, θ, v, ω], ...]
  """

  best_index = np.argmin(costs)
  best_trajectory = trajectories[best_index]
  return best_trajectory

# State Lattice Planner
def state_lattice_planner(current_state, goal_state, obstacles, distance_range, angle_range, 
                        num_distances, num_angles, params, time_horizon, dt):
  """
  State Lattice Planner を用いて経路を生成する

  Args:
      current_state: ロボットの現在状態 [x, y, θ, v, ω]
      goal_state: 目標状態 [x, y, θ, v, ω]
      obstacles: 障害物のリスト [[x, y, radius], ...]
      distance_range: サンプリングする距離の範囲 [min_distance, max_distance]
      angle_range: サンプリングする角度の範囲 [min_angle, max_angle]
      num_distances: サンプリングする距離の数
      num_angles: サンプリングする角度の数
      params: 入力ベクトル `u(p, t)` のパラメータベクトル `p`
      time_horizon: 時間軸
      dt: 時間刻み

  Returns:
      best_trajectory: 最適なパス [[x, y, θ, v, ω], ...]
  """

  # 1. 終端状態のサンプリング (Uniform Polar Sampling)
  terminal_states = uniform_polar_sampling(current_state, distance_range, angle_range, 
                                         num_distances, num_angles)

  # 2. パス生成 (Model Predictive Trajectory Generation)
  trajectories = []
  for terminal_state in terminal_states:
    # 目標状態の速度と角速度は現在の状態と同じにする
    terminal_state.extend(current_state[3:]) 
    trajectory = model_predictive_trajectory_generation(current_state, terminal_state, 
                                                     params, time_horizon, dt)
    trajectories.append(trajectory)

  # 3. 衝突チェックとコスト計算
  costs = []
  for trajectory in trajectories:
    if check_collision(trajectory, obstacles):
      costs.append(float('inf'))  # 衝突する場合はコストを無限大にする
    else:
      costs.append(calculate_cost(trajectory))

  # 4. パス選択
  best_trajectory = select_best_path(trajectories, costs)

  # 5. パラメータ最適化 (省略)

  return best_trajectory

# 障害物情報
obstacles = [[5, 5, 1], [10, 10, 2]]  # [[x, y, radius], ...]

# パラメータ
distance_range = [1, 5]
angle_range = [-math.pi / 4, math.pi / 4]
num_distances = 5
num_angles = 5
params = np.zeros(6)  # 入力ベクトル u(p, t) のパラメータベクトル p の初期値
time_horizon = np.arange(0, 2, 0.1)  # 時間軸
dt = 0.1  # 時間刻み

# ロボットの現在状態と目標状態
current_state = [0, 0, 0, 0, 0]  # [x, y, θ, v, ω]
goal_state = [10, 10, 0, 0, 0]

# 経路生成
best_trajectory = state_lattice_planner(current_state, goal_state, obstacles, distance_range, angle_range,
                                      num_distances, num_angles, params, time_horizon, dt)

print(f"最適なパス: {best_trajectory}")