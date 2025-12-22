import numpy as np

def gaussian_points_rotated(center=(0.0, 0.0), sigma=(1.0, 0.5), angle=0.0, n=100, seed=None):
    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=float)
    sx, sy = map(float, sigma)

    # sample axis-aligned
    pts = rng.normal(size=(n, 2)) * np.array([sx, sy])

    # rotate
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s],
                  [s,  c]])
    pts = pts @ R.T

    return pts + center

def gaussian_mixture(components, n_per=200, seed=None):
    """
    components: list of dicts like:
      {"center": (x,y), "sigma": 0.4}  OR
      {"center": (x,y), "cov": [[...],[...]]}
    Returns: (X, y)
      X shape (K*n_per, 2), y shape (K*n_per,)
    """
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for k, comp in enumerate(components):
        center = comp["center"]
        if "cov" in comp:
            Xk = rng.multivariate_normal(mean=center, cov=np.asarray(comp["cov"], float), size=n_per)
        else:
            sigma = comp.get("sigma", 1.0)
            Xk = rng.normal(size=(n_per, 2)) * float(sigma) + np.asarray(center, float)

        Xs.append(Xk)
        ys.append(np.full(n_per, k, dtype=int))

    return np.vstack(Xs), np.concatenate(ys)
