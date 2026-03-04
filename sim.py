import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rotation import rx, ry, rz
from params import Params
from matplotlib.animation import FuncAnimation


p = Params()
l1=0.2  
l2=0.2
d=0.05

p_hip = np.array([0, 0, 0.4]) # sqrt(0.05^2 + 0.05^2)
# Leg FK (hip 기준 foot 벡터)
def leg_fk(q):
    
    q1, q2, q3 = q
    shoulder = p_hip+rx(q1) @ np.array([0, d, 0])
    knee = shoulder + rx(q1) @ ry(q2) @ np.array([l1, 0, 0])
    foot = knee + rx(q1) @ ry(q2) @ ry(q3) @ np.array([l2, 0, 0])
    return shoulder, knee, foot

def leg_ik(p_des):

    # 1. hip 기준 벡터
    vp = p_des - p_hip

    # ----------------
    # q1
    # ----------------
    vpyz = vp[1:3]
    ryz = np.linalg.norm(vpyz)

    a = np.arcsin(vp[1]/ryz)
    b = np.arcsin(d/ryz)

    q1 = a - b   # sign_d=1 가정

    # ----------------
    # shoulder offset 제거
    # ----------------
    vd = np.array([0,
                   d*np.cos(q1),
                   d*np.sin(q1)])

    rf_vec = vp - vd
    rf = np.linalg.norm(rf_vec)

    # ----------------
    # q1 회전 제거
    # ----------------
    vf = rx(q1).T @ rf_vec

    # ----------------
    # q2
    # ----------------
    if vf[2] <= 0:
        a = np.arccos(vf[0]/rf)
    else:
        if vf[0] >= 0:
            a = -np.arcsin(vf[2]/rf)
        else:
            a = np.pi + np.arcsin(vf[2]/rf)

    cosb = (l1**2 + rf**2 - l2**2)/(2*l1*rf)
    b = np.arccos(cosb)

    q2 = a + b

    # ----------------
    # q3
    # ----------------
    cosc = (l1**2 + l2**2 - rf**2)/(2*l1*l2)
    c = np.arccos(cosc)

    q3 = -(np.pi - c)

    return np.array([q1,q2,q3])


q_init = np.array([0.0, 0.2, 0.0])
p_target = np.array([0.0, 0.05, 0.0])

q_sol = leg_ik(p_target)

print("IK solution:", q_sol)

# FK with solution
shoulder,knee,foot = leg_fk(q_sol)

def foot_trajectory(t, T):
    step_length = 0.15
    step_height = 0.08

    phase = (t % T) / T   # 0~1

    x = -step_length/2 + step_length * phase
    y = 0.05
    z = 0.0 + step_height * np.sin(np.pi * phase)

    return np.array([x, y, z])
# --------------------------------
# Pre-compute reference trajectory
# --------------------------------
T=1.0
t_samples = np.linspace(0, T, 200)
traj = np.array([foot_trajectory(t, T) for t in t_samples])
# ------------------------
# Plot
# ------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 🔴 미리 정의된 궤적
ax.plot(traj[:,0], traj[:,1], traj[:,2],
        'r', linewidth=2, label="Foot Trajectory")
line, = ax.plot([], [], [], '-o', lw=3)
target_point = ax.scatter([], [], [], color='r')

ax.set_xlim([-0.3,0.3])
ax.set_ylim([-0.3,0.3])
ax.set_zlim([0,0.6])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

T = 1.0   # 1초 주기

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

def update(frame):

    t = frame * 0.02
    p_target = foot_trajectory(t, T)

    try:
        q_sol = leg_ik(p_target)
        shoulder, knee, foot = leg_fk(q_sol)

        points = np.vstack([p_hip, shoulder, knee, foot])

        line.set_data(points[:,0], points[:,1])
        line.set_3d_properties(points[:,2])

        ax.collections.clear()
        ax.scatter(*p_target, color='r')

    except:
        pass

    return line,

ani = FuncAnimation(fig, update,
                    frames=300,
                    init_func=init,
                    interval=20,
                    blit=False)

plt.show()
