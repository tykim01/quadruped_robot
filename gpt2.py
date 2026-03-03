import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rotation import rx, ry, rz
from params import Params

p = Params()

# =========================
# Rotation
# =========================
def rpy(roll, pitch, yaw=0.0):
    return rz(yaw) @ ry(pitch) @ rx(roll)

# =========================
# Leg FK (hip 기준 foot 벡터)
# =========================
def leg_fk_vector(q, l1, l2, d):
    q1, q2, q3 = q
    shoulder = rx(q1) @ np.array([0, d, 0])
    knee = shoulder + rx(q1) @ ry(q2) @ np.array([l1, 0, 0])
    foot = knee + rx(q1) @ ry(q2) @ ry(q3) @ np.array([l2, 0, 0])
    return foot

# =========================
# Floating base solve
# =========================
def solve_body_pose(qs):

    hip_offsets = {
        "RL": np.array([0,0,0]),
        "RR": np.array([0,-p.W,0]),
        "FL": np.array([p.L,0,0]),
        "FR": np.array([p.L,-p.W,0])
    }

    feet_body = {}

    for leg in qs:
        d = p.d if leg in ["RL","FL"] else -p.d
        feet_body[leg] = hip_offsets[leg] + \
            leg_fk_vector(qs[leg], p.l1, p.l2, d)

    def equations(x):
        roll, pitch, z = x
        R = rpy(roll, pitch)
        body_pos = np.array([0,0,z])

        eq = []
        for leg in feet_body:
            foot_world = body_pos + R @ feet_body[leg]
            eq.append(foot_world[2])

        return eq[:3]  # 과구속 방지

    sol = fsolve(equations, [0,0,0.3])
    return sol

# =========================
# Draw Body
# =========================
def draw_body(ax, body_pos, R):

    L, W, H = p.L, p.W, p.h

    vertices_body = np.array([
        [0,0,0],
        [L,0,0],
        [L,-W,0],
        [0,-W,0],
        [0,0,H],
        [L,0,H],
        [L,-W,H],
        [0,-W,H],
    ])

    vertices_world = body_pos + (R @ vertices_body.T).T

    faces = [
        [vertices_world[0], vertices_world[1], vertices_world[2], vertices_world[3]],
        [vertices_world[4], vertices_world[5], vertices_world[6], vertices_world[7]],
        [vertices_world[0], vertices_world[1], vertices_world[5], vertices_world[4]],
        [vertices_world[1], vertices_world[2], vertices_world[6], vertices_world[5]],
        [vertices_world[2], vertices_world[3], vertices_world[7], vertices_world[6]],
        [vertices_world[3], vertices_world[0], vertices_world[4], vertices_world[7]],
    ]

    body = Poly3DCollection(
        faces,
        facecolor='blue',
        edgecolor='black',
        alpha=0.9
    )

    ax.add_collection3d(body)

# =========================
# Main
# =========================
if __name__ == "__main__":

    qs = {
        "RL": np.array([0.0, 3*np.pi/4, -3*np.pi/4]),
        "RR": np.array([0.0, 3*np.pi/4, -3*np.pi/4]),
        "FL": np.array([0.0, 3*np.pi/4, -2*np.pi/4]),
        "FR": np.array([0.0, 3*np.pi/4, -2*np.pi/4]),
    }

    roll, pitch, z = solve_body_pose(qs)

    print("roll :", roll)
    print("pitch:", pitch)

    R = rpy(roll, pitch)
    body_pos = np.array([0,0,z])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    hip_offsets = {
        "RL": np.array([0,0,0]),
        "RR": np.array([0,-p.W,0]),
        "FL": np.array([p.L,0,0]),
        "FR": np.array([p.L,-p.W,0])
    }

    for leg in qs:
        d = p.d if leg in ["RL","FL"] else -p.d
        q = qs[leg]

        hip = hip_offsets[leg]
        shoulder = hip + rx(q[0]) @ np.array([0,d,0])
        knee = shoulder + rx(q[0]) @ ry(q[1]) @ np.array([p.l1,0,0])
        foot = hip + leg_fk_vector(q, p.l1, p.l2, d)

        points = np.vstack((hip, shoulder, knee, foot))
        world = body_pos + (R @ points.T).T

        ax.plot(*world.T, marker='o', linewidth=3)

    # 🔵 body 면 그리기
    draw_body(ax, body_pos, R)

    # 지면 표시
    ax.plot([-1,1], [0,0], [0,0], alpha=0)

    ax.set_xlim(-0.2,0.6)
    ax.set_ylim(-0.4,0.2)
    ax.set_zlim(0,0.8)
    ax.set_box_aspect([1,1,1])

    plt.show()