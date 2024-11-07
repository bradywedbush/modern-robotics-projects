# %%
import numpy as np
import modern_robotics as mr


# %%
def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame for an open chain robot,
        and prints the iterates to output and creates a csv for each iteration's
        joint vector

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    :return iterates: A text output containing all iterations and their joint
                      vector, end effector config, error-twist, and angular /
                      linear errors
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
        Iteration 3:
        joint vector:
        0.221, 0.375, 2.233, 1.414
        SE(3) end-effector config:
        1.000 0.000 0.000 3.275
        0.000 1.000 0.000 4.162
        0.000 0.000 1.000 -5.732
        0     0     0     1
        error twist V_b: (0.232, 0.171, 0.211, 0.345, 1.367, -0.222)
        angular error ||omega_b||: 0.357
        linear error ||v_b||: 1.427
    """

    # Create csv to save to
    # Overwrite
    f = open("output_p2_short.csv", "w")

    # Newton-Raphson method initial config
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 50
    Tsb = mr.FKinBody(M, Blist, thetalist)
    Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tsb), T)))
    Vb_arr = [Vb]
    omega_b = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
    v_b = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
    err = omega_b > eomg or v_b > ev

    f.write(
        f"{thetalist[0]: .3f}, {thetalist[1]: .3f}, {thetalist[2]: .3f}, {thetalist[3]: .3f}, {thetalist[4]: .3f}, {thetalist[5]: .3f}\n"
    )

    # Print iteration zero
    np.set_printoptions(formatter={"float": lambda x: "{0: 0.3f}".format(x)})
    print(f"Iteration {i}:\n")
    print(f"joint vector:\n{thetalist}\n")
    print(f"SE(3) end-effector config: \n{Tsb}\n")
    print(f"error twist V_b: {Vb}")
    print(f"angular error ||omega_b||: {omega_b: .4f}")
    print(f"linear error ||v_b||: {v_b: .4f}\n\n")

    # Newton-Raphson method iterations
    while err and i < maxiterations:
        i = i + 1
        thetalist = thetalist + np.dot(
            np.linalg.pinv(mr.JacobianBody(Blist, thetalist)), Vb
        )
        Tsb = mr.FKinBody(M, Blist, thetalist)
        Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tsb), T)))
        Vb_arr.append(Vb)
        omega_b = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
        v_b = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
        err = omega_b > eomg or v_b > ev

        # Print iteration i
        print(f"Iteration {i}:\n")
        print(f"joint vector:\n{thetalist}\n")
        print(f"SE(3) end-effector config: \n{Tsb}\n")
        print(f"error twist V_b: {Vb}")
        print(f"angular error ||omega_b||: {omega_b: .4f}")
        print(f"linear error ||v_b||: {v_b: .4f}\n\n")

        # Write to csv file
        thetalist = thetalist % (2 * np.pi)
        f.write(
            f"{thetalist[0]: .3f}, {thetalist[1]: .3f}, {thetalist[2]: .3f}, {thetalist[3]: .3f}, {thetalist[4]: .3f}, {thetalist[5]: .3f}\n"
        )

    f.close()
    return (thetalist, not err)


# %%
# Main block
# Ch 4.1.2 Fig 4.6

L1 = 425 / 1000.0
L2 = 392 / 1000.0
W1 = 109 / 1000.0
W2 = 82 / 1000.0
H1 = 89 / 1000.0
H2 = 95 / 1000.0

Blist = np.array(
    [
        [0, 1, 0, W1 + W2, 0, L1 + L2],
        [0, 0, 1, H2, -L1 - L2, 0],
        [0, 0, 1, H2, -L2, 0],
        [0, 0, 1, H2, 0, 0],
        [0, -1, 0, -W2, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ]
).T

Tsd = np.array(
    [
        [0.7071, 0, 0.7071, -0.3],
        [0.7071, 0, -0.7071, -0.5],
        [0, 1, 0, 0.5],
        [0, 0, 0, 1],
    ]
)

Mi = np.array(
    [[-1, 0, 0, L1 + L2], [0, 0, 1, W1 + W2], [0, 1, 0, H1 - H2], [0, 0, 0, 1]]
)

eomg = 0.001
ev = 0.0001

# Long iterates theta0 -- change filename in function
# theta0 = np.array([0, 0, 0, 0, 0, 0])

# Short iterates theta0 -- change filename in function
theta0 = np.array([3, -1, 1, 0, 0, 0])

print(IKinBodyIterates(Blist, Mi, Tsd, theta0, eomg, ev))

# Short iterates prints
# (array([ 3.839,  5.167,  0.795,  0.323,  6.195,  6.283]), True)
# in 3 iterations

# Long iterates prints
# (array([ 1.083,  4.293,  4.946,  0.185,  2.844,  3.142]), True)
# in 23 iterations
# This answer is the flipped answer of short iterates
