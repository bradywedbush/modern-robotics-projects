# %%
import numpy as np
import modern_robotics as mr

# %%
# Import rotation matrices (given)
r13 = np.array([[-0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, -0.7071]])
rs2 = np.array([[-0.6964, 0.1736, 0.6964], [-0.1228, -0.9848, 0.1228], [0.7071, 0, 0.7071]])
r25 = np.array([[-0.7566, -0.1198, -0.6428], [-0.1564, 0.9877, 0], [0.6348, 0.1005, -0.7661]])
r12 = np.array([[0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, 0.7071]])
r34 = np.array([[0.6428, 0, -0.7660], [0, 1, 0], [0.7660, 0, 0.6428]])
rs6 = np.array([[0.9418, 0.3249, -0.0859], [0.3249, -0.9456, -0.0151], [-0.0861, -0.0136, -0.9962]])
r6b = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

# Rotation axis for each joint -- initial configuration (given)
omg_targ = (np.array([0, 0,  1]), 
            np.array([0, 1,  0]), 
            np.array([0, 1,  0]), 
            np.array([0, 1,  0]),
            np.array([0, 0, -1]),
            np.array([0, 1,  0]))

# Calculate necessary matrices
rs1 = mr.ProjectToSO3(rs2 @ r12.T)
r12 = mr.ProjectToSO3(r12)
r23 = mr.ProjectToSO3(r12.T @ r13)
r34 = mr.ProjectToSO3(r34)
r45 = mr.ProjectToSO3(r34.T @ r23.T @ r25)
r56 = mr.ProjectToSO3(r25.T @ rs2.T @ rs6)

print("Rsb = ")
print(rs6 @ r6b)

Rsto6 = (rs1, r12, r23, r34, r45, r56)
n = len(Rsto6)

E = 1E-4
i = 0
print("Joint angle inputs:")
for r in Rsto6 :
    #Acquire joint angles using MR library
    so3mat = mr.MatrixLog3(r)
    omega_theta = mr.so3ToVec(so3mat)
    omg, theta = mr.AxisAng3(omega_theta)
    omg = np.round(omg)
    # determine if omg from AxisAng3 function is different from the omg_target
    if np.linalg.norm(omg + omg_targ[i]) <= 1 : omg, theta = (np.absolute(omg), -theta)

    # print in csv format with no comma and a new line at the end
    s = ", " * (i!=(n-1)) + "/n/n" * (i==(n-1))
    print(f"{theta: .3f}", end= s)

    i += 1

    # DEBUGGING
    # print(f'Omega {i} = {omg}')
    # print(f'Theta {i} = {theta: .2f} rad')
