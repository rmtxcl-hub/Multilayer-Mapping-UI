import numpy as np

TINY = 1e-300

def _grid(x, N=501):
    if isinstance(x, str):
        s = x.strip().lower().replace(" ", "")
        if "-" in s:
            p = s.split("-")
            a, b = float(p[0]), float(p[1])
            n = int(p[2]) if len(p) >= 3 else N
            return np.linspace(a, b, max(n, 2))
        return np.array([float(s)], float)
    if np.isscalar(x):
        return np.array([float(x)], float)
    return np.asarray(x, float).ravel()

def _lam(cfg, lam=None):
    return np.asarray(_grid(cfg["lam_um"] if lam is None else lam, int(cfg.get("lamN", 501))), float)

def _theta_deg(cfg, theta=None):
    if theta is None:
        return np.asarray(cfg["theta_rad"], float).ravel() * 180.0 / np.pi
    return _grid(theta, int(cfg.get("thN", 301)))

def _theta_rad(th_deg):
    return np.deg2rad(np.asarray(th_deg, float))

def _nk_stack(cfg, lam_um, nk=None):
    n0, dL = cfg["n0"], list(map(float, cfg["dL"]))
    if nk is not None:
        ns = nk.nk(cfg["sub_key"], lam_um)
        nL = [nk.nk(k, lam_um) for k in cfg["film_keys"]]
    else:
        ns, nL = cfg["ns"], cfg["nL"]
        if np.asarray(ns).shape[0] != lam_um.size:
            raise ValueError("cfg nk grid != lam grid; pass nk=NK() or match wavelength grids")
    if len(nL) != len(dL):
        raise ValueError("len(nL) != len(dL)")
    return n0, np.asarray(ns, complex), [np.asarray(nj, complex) for nj in nL], dL

def _exp_pm(delta, clip=60.0):
    a = np.real(delta)
    b = np.clip(np.imag(delta), -float(clip), float(clip))
    return np.exp(1j * a - b), np.exp(-1j * a + b)

def _state_one_theta(tmm, lam_um, theta_rad, n0, ns, nL, dL, pol):
    s0 = np.sin(float(theta_rad))
    c0 = tmm.ct(n0, s0, n0)
    q0 = tmm.q(n0, c0, pol)
    r = tmm.r(lam_um, float(theta_rad), n0, ns, nL, dL, pol)

    E = 1.0 + r
    H = q0 * (1.0 - r)

    Ep, Em, Hp, Hm, qL, cL, sL, Et, Ht = [], [], [], [], [], [], [], [], []

    for nj, dj in zip(nL, dL):
        cj = tmm.ct(n0, s0, nj)
        qj = tmm.q(nj, cj, pol)
        sj = (n0 * s0) / nj

        ep = 0.5 * (E + H / qj)
        em = 0.5 * (E - H / qj)

        Ep.append(ep.copy())
        Em.append(em.copy())
        Hp.append((qj * ep).copy())
        Hm.append((-qj * em).copy())
        qL.append(qj.copy())
        cL.append(cj.copy())
        sL.append(sj.copy())
        Et.append(E.copy())
        Ht.append(H.copy())

        delta = (2.0 * np.pi / lam_um) * nj * float(dj) * cj
        cd, sd = tmm.cs(delta)
        E, H = cd * E + 1j * sd * H / qj, 1j * qj * sd * E + cd * H

    return {
        "r": r,
        "q0": q0,
        "E_plus": np.asarray(Ep),
        "E_minus": np.asarray(Em),
        "H_plus": np.asarray(Hp),
        "H_minus": np.asarray(Hm),
        "q": np.asarray(qL),
        "cos_theta": np.asarray(cL),
        "sin_theta": np.asarray(sL),
        "E_top": np.asarray(Et),
        "H_top": np.asarray(Ht),
    }

def amplitudes(tmm, cfg, lam=None, theta=None, pol=None, nk=None):
    pol = str(cfg.get("pol") if pol is None else pol).lower()
    if pol == "both":
        raise ValueError("amplitudes() requires pol='s' or pol='p', not 'both'")

    lam_um = _lam(cfg, lam)
    th_deg = _theta_deg(cfg, theta)
    th_rad = _theta_rad(th_deg)
    n0, ns, nL, dL = _nk_stack(cfg, lam_um, nk=nk)

    nl, nw, nt = len(dL), lam_um.size, th_rad.size

    out = {
        "E_plus": np.empty((nl, nw, nt), complex),
        "E_minus": np.empty((nl, nw, nt), complex),
        "H_plus": np.empty((nl, nw, nt), complex),
        "H_minus": np.empty((nl, nw, nt), complex),
        "q": np.empty((nl, nw, nt), complex),
        "cos_theta": np.empty((nl, nw, nt), complex),
        "sin_theta": np.empty((nl, nw, nt), complex),
        "E_top": np.empty((nl, nw, nt), complex),
        "H_top": np.empty((nl, nw, nt), complex),
        "r": np.empty((nw, nt), complex),
        "q0": np.empty((nw, nt), complex),
    }

    for it, th in enumerate(th_rad):
        st = _state_one_theta(tmm, lam_um, th, n0, ns, nL, dL, pol)
        for k in ("E_plus", "E_minus", "H_plus", "H_minus", "q", "cos_theta", "sin_theta", "E_top", "H_top"):
            out[k][:, :, it] = st[k]
        out["r"][:, it] = st["r"]
        out["q0"][:, it] = st["q0"]

    out.update({
        "lam_um": lam_um,
        "theta_deg": th_deg,
        "theta_rad": th_rad,
        "pol": pol,
        "dL": dL,
        "film_keys": cfg.get("film_keys"),
        "sub_key": cfg.get("sub_key"),
    })

    return out

def _components_from_amplitudes(lam_um, z_um, ep0, em0, qj, cj, sj, nj, pol, clip=60.0):
    delta = (2.0 * np.pi / lam_um) * nj * float(z_um) * cj
    fp, fm = _exp_pm(delta, clip)
    ep, em = ep0 * fp, em0 * fm
    z0 = np.zeros_like(ep, complex)

    if pol == "s":
        Ey = ep + em
        Hx = -qj * (ep - em)
        Hz = nj * sj * (ep + em)
        return {"Ex": z0, "Ey": Ey, "Ez": z0, "Hx": Hx, "Hy": z0, "Hz": Hz}

    Ex = ep + em
    Ez = -(sj / cj) * ep + (sj / cj) * em
    Hy = qj * (ep - em)
    return {"Ex": Ex, "Ey": z0, "Ez": Ez, "Hx": z0, "Hy": Hy, "Hz": z0}

def components_at(tmm, cfg, layer=1, z_um=None, z_frac=0.5, lam=None, theta=None, pol=None, nk=None):
    pol = str(cfg.get("pol") if pol is None else pol).lower()
    if pol == "both":
        raise ValueError("components_at() requires pol='s' or pol='p', not 'both'")

    lam_um = _lam(cfg, lam)
    n0, ns, nL, dL = _nk_stack(cfg, lam_um, nk=nk)
    amp = amplitudes(tmm, cfg, lam=lam_um, theta=theta, pol=pol, nk=nk)

    j = int(layer) - 1
    if j < 0 or j >= len(dL):
        raise ValueError("layer must be 1-based and within the film stack")

    z = float(z_frac) * float(dL[j]) if z_um is None else float(z_um)
    nw, nt = amp["lam_um"].size, amp["theta_deg"].size

    comps = {k: np.empty((nw, nt), complex) for k in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}

    for it in range(nt):
        c = _components_from_amplitudes(
            amp["lam_um"],
            z,
            amp["E_plus"][j, :, it],
            amp["E_minus"][j, :, it],
            amp["q"][j, :, it],
            amp["cos_theta"][j, :, it],
            amp["sin_theta"][j, :, it],
            nL[j],
            pol,
            getattr(tmm, "clip_im_delta", 60.0),
        )
        for k in comps:
            comps[k][:, it] = c[k]

    comps.update({
        "lam_um": amp["lam_um"],
        "theta_deg": amp["theta_deg"],
        "theta_rad": amp["theta_rad"],
        "layer": j + 1,
        "z_um": z,
        "pol": pol,
        "q0": amp["q0"],
        "r": amp["r"],
    })

    return comps

def intensity_from_components(comps, field="E", component="total", normalization="incident"):
    field = str(field).strip().upper()
    component = str(component).strip().lower()
    normalization = str(normalization).strip().lower()

    names = ("Ex", "Ey", "Ez") if field == "E" else ("Hx", "Hy", "Hz")
    a, b, c = [comps[k] for k in names]

    if component in ("x", "ex", "hx"):
        val = np.abs(a) ** 2
    elif component in ("y", "ey", "hy"):
        val = np.abs(b) ** 2
    elif component in ("z", "ez", "hz", "normal"):
        val = np.abs(c) ** 2
    elif component in ("t", "tan", "tangential"):
        val = np.abs(a) ** 2 + np.abs(b) ** 2
    else:
        val = np.abs(a) ** 2 + np.abs(b) ** 2 + np.abs(c) ** 2

    if field == "H" and not normalization.startswith("raw"):
        val = val / np.maximum(np.abs(comps["q0"]) ** 2, TINY)

    if normalization in ("max", "map max", "map_max", "normalized 0-1"):
        m = np.nanmax(val[np.isfinite(val)]) if np.any(np.isfinite(val)) else 0.0
        val = val / m if m > 0 else val

    if normalization in ("log", "log10", "log10 intensity"):
        val = np.log10(np.maximum(val, TINY))

    return val

def intensity_at(tmm, cfg, field="E", component="total", layer=1, z_um=None, z_frac=0.5, lam=None, theta=None, pol=None, nk=None, normalization="incident"):
    comps = components_at(tmm, cfg, layer=layer, z_um=z_um, z_frac=z_frac, lam=lam, theta=theta, pol=pol, nk=nk)
    val = intensity_from_components(comps, field=field, component=component, normalization=normalization)
    return {
        "val": val,
        "lam_um": comps["lam_um"],
        "theta_deg": comps["theta_deg"],
        "theta_rad": comps["theta_rad"],
        "layer": comps["layer"],
        "z_um": comps["z_um"],
        "field": field,
        "component": component,
        "normalization": normalization,
        "pol": comps["pol"],
        "r": comps["r"],
    }