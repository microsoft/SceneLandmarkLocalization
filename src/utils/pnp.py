import numpy as np
from scipy.optimize import least_squares


def Rotation2Quaternion(R):
    """
    Convert a rotation matrix to quaternion

    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix
    Returns
    -------
    q : ndarray of shape (4,)
        The unit quaternion (w, x, y, z)
    """
    q = np.empty([4, ])

    tr = np.trace(R)
    if tr < 0:
        i = R.diagonal().argmax()
        j = (i + 1) % 3
        k = (j + 1) % 3

        q[i] = np.sqrt(1 - tr + 2 * R[i, i]) / 2
        q[j] = (R[j, i] + R[i, j]) / (4 * q[i])
        q[k] = (R[k, i] + R[i, k]) / (4 * q[i])
        q[3] = (R[k, j] - R[j, k]) / (4 * q[i])
    else:
        q[3] = np.sqrt(1 + tr) / 2
        q[0] = (R[2, 1] - R[1, 2]) / (4 * q[3])
        q[1] = (R[0, 2] - R[2, 0]) / (4 * q[3])
        q[2] = (R[1, 0] - R[0, 1]) / (4 * q[3])

    q /= np.linalg.norm(q)
    # Rearrange (x, y, z, w) to (w, x, y, z)
    q = q[[3, 0, 1, 2]]

    return q


def Quaternion2Rotation(q):
    """
    Convert a quaternion to rotation matrix

    Parameters
    ----------
    q : ndarray of shape (4,)
        Unit quaternion (w, x, y, z)
    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    """
    q /= np.linalg.norm(q)

    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    R = np.empty([3, 3])
    R[0, 0] = 1 - 2 * y ** 2 - 2 * z ** 2
    R[0, 1] = 2 * (x * y - z * w)
    R[0, 2] = 2 * (x * z + y * w)

    R[1, 0] = 2 * (x * y + z * w)
    R[1, 1] = 1 - 2 * x ** 2 - 2 * z ** 2
    R[1, 2] = 2 * (y * z - x * w)

    R[2, 0] = 2 * (x * z - y * w)
    R[2, 1] = 2 * (y * z + x * w)
    R[2, 2] = 1 - 2 * x ** 2 - 2 * y ** 2

    return R

def skewsymm(x):
    Sx = np.zeros((3, 3))
    Sx[0, 1] = -x[2]
    Sx[0, 2] = x[1]
    Sx[1, 0] = x[2]
    Sx[2, 0] = -x[1]
    Sx[1, 2] = -x[0]
    Sx[2, 1] = x[0]

    return Sx


def SetupPnPNL(C_T_G, G_p_f, C_b_f):
    n_points = G_p_f.shape[1]
    n_projs = n_points
    b = np.zeros((3 * n_projs,))

    for k in range(n_points):
        b[3 * k: 3 * (k + 1)] = C_b_f[:, k]

    z = np.zeros((7 + 3 * n_points,))
    R = C_T_G[:3, :3]
    t = C_T_G[:3, 3]
    q = Rotation2Quaternion(R)

    z[0:7] = np.concatenate([t, q])
    for i in range(n_points):
        z[7 + 3 * i: 7 + 3 * (i + 1)] = G_p_f[:, i]

    return z, b


def MeasureReprojectionSinglePose(z, b, p, w):

    n_projs = b.shape[0] // 3
    f = np.zeros((3 * n_projs,))
    s = 0.001 * np.ones((3 * n_projs,))

    q = z[3:7]
    q_norm = np.sqrt(np.sum(q ** 2))
    q = q / q_norm
    R = Quaternion2Rotation(q)
    t = z[:3]

    for j in range(n_projs):
        X = p[3 * j:3 * (j + 1)]
        # Remove measurement error of fixed poses
        b_hat = R @ X + t
        f[3 * j: 3 * (j + 1)] = b_hat / np.sqrt(np.sum(b_hat ** 2))
        s[3 * j: 3 * (j + 1)] = w[j]

    err = s * (b - f)

    return err


def UpdatePose(z):

    p = z[0:7]
    q = p[3:]

    q = q / np.linalg.norm(q)
    R = Quaternion2Rotation(q)
    t = p[:3]
    P_new = np.hstack([R, t[:, np.newaxis]])

    return P_new


def P3PKe(m, X, inlier_thres=1e-5):

    w1 = X[:, 0]
    w2 = X[:, 1]
    w3 = X[:, 2]

    u0 = w1 - w2
    nu0 = np.linalg.norm(u0)
    if nu0 < 1e-4:
        return None, None
    k1 = u0 / nu0

    b1 = m[:, 0]
    b2 = m[:, 1]
    b3 = m[:, 2]

    k3 = np.cross(b1, b2)
    nk3 = np.linalg.norm(k3)
    if nk3 < 1e-4:
        return None, None
    k3 = k3 / nk3

    tz = np.cross(b1, k3)
    v1 = np.cross(b1, b3)
    v2 = np.cross(b2, b3)

    u1 = w1 - w3
    u1k1 = np.sum(u1 * k1)
    k3b3 = np.sum(k3 * b3)
    if np.abs(k3b3) < 1e-4:
        return None, None


    f11 = k3.T @ b3
    f13 = k3.T @ v1
    f15 = -u1k1 * f11
    nl = np.cross(u1, k1)
    delta = np.linalg.norm(nl)
    if delta < 1e-4:
        return None, None
    nl = nl / delta
    f11 = delta * f11
    f13 = delta * f13

    u2k1 = u1k1 - nu0
    f21 = np.sum(tz * v2)
    f22 = nk3 * k3b3
    f23 = np.sum(k3 * v2)
    f24 = u2k1 * f22
    f25 = -u2k1 * f21
    f21 = delta * f21
    f22 = delta * f22
    f23 = delta * f23

    g1 = f13 * f22
    g2 = f13 * f25 - f15 * f23
    g3 = f11 * f23 - f13 * f21
    g4 = -f13 * f24
    g5 = f11 * f22
    g6 = f11 * f25 - f15 * f21
    g7 = -f15 * f24
    alpha = np.array([g5 * g5 + g1 * g1 + g3 * g3,
                      2 * (g5 * g6 + g1 * g2 + g3 * g4),
                      g6 * g6 + 2 * g5 * g7 + g2 * g2 + g4 * g4 - g1 * g1 - g3 * g3,
                      2 * (g6 * g7 - g1 * g2 - g3 * g4),
                      g7 * g7 - g2 * g2 - g4 * g4])

    if any(np.isnan(alpha)):
        return None, None

    sols = np.roots(alpha)

    Ck1nl = np.vstack((k1, nl, np.cross(k1, nl))).T
    Cb1k3tzT = np.vstack((b1, k3, tz))
    b3p = (delta / k3b3) * b3

    R = np.zeros((3, 3, 4))
    t = np.zeros((3, 4))
    for i in range(sols.shape[0]):
        if np.imag(sols[i]) != 0:
            continue

        ctheta1p = np.real(sols[i])
        if abs(ctheta1p) > 1:
            continue
        stheta1p = np.sqrt(1 - ctheta1p * ctheta1p)
        if k3b3 < 0:
            stheta1p = -stheta1p

        ctheta3 = g1 * ctheta1p + g2
        stheta3 = g3 * ctheta1p + g4
        ntheta3 = stheta1p / ((g5 * ctheta1p + g6) * ctheta1p + g7)
        ctheta3 = ntheta3 * ctheta3
        stheta3 = ntheta3 * stheta3

        C13 = np.array([[ctheta3, 0, -stheta3],
                        [stheta1p * stheta3, ctheta1p, stheta1p * ctheta3],
                        [ctheta1p * stheta3, -stheta1p, ctheta1p * ctheta3]])

        Ri = (Ck1nl @ C13 @ Cb1k3tzT).T
        pxstheta1p = stheta1p * b3p
        ti = pxstheta1p - Ri @ w3
        ti = ti.reshape(3, 1)

        m_hat = Ri @ X + ti
        m_hat = m_hat / np.linalg.norm(m_hat, axis=0)
        if np.sum(np.sum(m_hat * m, axis=0) > 1.0 - inlier_thres) == 4:
            return Ri, ti

    return None, None


def P3PKe_Ransac(G_p_f, C_b_f_hm, w, thres=0.01):
    inlier_thres = thres
    C_T_G_best = None
    inlier_best = np.zeros(G_p_f.shape[1], dtype=bool)
    Nsample=4
    inlier_score_best=0

    for iter in range(50):

        ## uniform sampling based on weight factor
        # min_set = np.argpartition(np.exp(w * np.random.randn(w.shape[0])), -Nsample)[-Nsample:]
        # min_set = np.argpartition(w * np.random.rand(w.shape[0]), -Nsample)[-Nsample:]
        min_set = np.argpartition(np.exp(10.0 * w) * np.random.rand(w.shape[0]), -Nsample)[-Nsample:]

        C_R_G_hat, C_t_G_hat = P3PKe(C_b_f_hm[:, min_set], G_p_f[:, min_set], inlier_thres=thres)

        if C_R_G_hat is None or C_t_G_hat is None:
            continue

        # Get inlier
        C_b_f_hat = C_R_G_hat @ G_p_f + C_t_G_hat
        C_b_f_hat = C_b_f_hat / np.linalg.norm(C_b_f_hat, axis=0)
        inlier_mask = np.sum(C_b_f_hat * C_b_f_hm, axis=0) > (1.0 - inlier_thres)
        inlier_score = np.sum(w[inlier_mask])
        # inlier_score = np.sum(inlier_mask)
        if inlier_score > inlier_score_best:
            inlier_best = inlier_mask
            C_T_G_best = np.eye(4)
            C_T_G_best[:3, :3] = C_R_G_hat
            C_T_G_best[:3, 3:] = C_t_G_hat
            inlier_score_best = inlier_score

    return C_T_G_best, inlier_best


def RunPnPNL(C_T_G, G_p_f, C_b_f, w, cutoff=0.01):

    z0, b = SetupPnPNL(C_T_G, G_p_f, C_b_f)
    res = least_squares(
        lambda x: MeasureReprojectionSinglePose(x, b, z0[7:], w),
        z0[:7],
        verbose=0,
        ftol=1e-4,
        max_nfev=50,
        xtol=1e-8,
        loss='huber',
        f_scale=cutoff
    )
    z = res.x

    P_new = UpdatePose(z)

    return P_new


