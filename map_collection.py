import numpy as np
import map_mod as mm

C1_RAD_UM = 1.191042972e8
C2_UM_K = 1.438776877e4

MAP_ALIASES = {
    "r": "reflectivity",
    "reflectivity": "reflectivity",
    "e": "emissivity",
    "emissivity": "emissivity",
    "band-averaged emissivity": "band_avg_emissivity",
    "band averaged emissivity": "band_avg_emissivity",
    "planck-weighted emissive power": "planck_power",
    "planck weighted emissive power": "planck_power",
    "hemispherical emissivity": "hemispherical_emissivity",
    "polarization contrast": "polarization_contrast",
    "ellipsometry": "ellipsometry",
}

CONTRAST_ALIASES = {
    "eps norm": "eps_norm",
    "eps normalized": "eps_norm",
    "(εₚ-εₛ)/(εₚ+εₛ)": "eps_norm",
    "(εp-εs)/(εp+εs)": "eps_norm",
    "eps diff": "eps_diff",
    "εₚ-εₛ": "eps_diff",
    "εp-εs": "eps_diff",
    "r norm": "r_norm",
    "(rₚ-rₛ)/(rₚ+rₛ)": "r_norm",
    "(rp-rs)/(rp+rs)": "r_norm",
    "r diff": "r_diff",
    "rₚ-rₛ": "r_diff",
    "rp-rs": "r_diff",
}

ELLIP_ALIASES = {
    "ψ angle (deg)": "psi_deg",
    "psi deg": "psi_deg",
    "ψ deg": "psi_deg",
    "ψ angle (rad)": "psi_rad",
    "psi rad": "psi_rad",
    "δ phase (deg)": "delta_deg",
    "delta deg": "delta_deg",
    "δ deg": "delta_deg",
    "ρ phase deg": "delta_deg",
    "rho phase deg": "delta_deg",
    "arg ρ deg": "delta_deg",
    "arg rho deg": "delta_deg",
    "δ phase (rad)": "delta_rad",
    "delta rad": "delta_rad",
    "|ρ| amplitude": "rho_abs",
    "|ρ|": "rho_abs",
    "|rho|": "rho_abs",
    "rho abs": "rho_abs",
    "tan ψ": "rho_abs",
    "tan psi": "rho_abs",
}

def _trapz(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    if hasattr(np, "trapz"):
        return np.trapz(y, x, axis=axis)
    from scipy.integrate import trapezoid
    return trapezoid(y, x, axis=axis)


def _mode_key(mode):
    s = str(mode).strip().lower()
    return MAP_ALIASES.get(s, s.replace(" ", "_").replace("-", "_"))

def _grid(x, N=501):
    return mm._grid(x, N=N)

def _lam(cfg, lam=None):
    return mm._um_fix(_grid(cfg["lam_um"] if lam is None else lam))

def _theta_deg(cfg, theta=None):
    src = cfg["theta_rad"] * 180 / np.pi if theta is None else theta
    return mm._th_deg(src)

def _theta_rad(th_deg):
    return mm._th_rad(th_deg)

def _eps_from_r(r):
    return 1.0 - np.abs(r) ** 2

def _pack_meta(cfg, extra=None):
    meta = dict(
        n0=cfg.get("n0"),
        film_keys=cfg.get("film_keys"),
        sub_key=cfg.get("sub_key"),
        dL=list(cfg.get("dL", [])),
        cover_n=cfg.get("cover_n"),
        cover_k=cfg.get("cover_k"),
    )
    if extra:
        meta.update(extra)
    return meta

def _nk_stack(cfg, lam_um, nk=None):
    n0, dL = cfg["n0"], cfg["dL"]
    if nk is not None:
        ns = nk.nk(cfg["sub_key"], lam_um)
        nL = [nk.nk(k, lam_um) for k in cfg["film_keys"]]
    else:
        ns, nL = cfg["ns"], cfg["nL"]
        if np.asarray(ns).shape[0] != lam_um.size:
            raise ValueError("cfg nk grid != map lam; pass nk=NK() or match lam grids")
    return n0, ns, nL, dL

def _rs_rp(tmm, cfg, lam_um, th_rad, nk=None):
    n0, ns, nL, dL = _nk_stack(cfg, lam_um, nk=nk)
    rs = tmm.r(lam_um, th_rad, n0, ns, nL, dL, "s")
    rp = tmm.r(lam_um, th_rad, n0, ns, nL, dL, "p")
    return rs, rp

def _base_eps(tmm, cfg, lam_um, th_rad, pol, nk=None):
    pol = str(pol).lower()
    if pol == "both":
        rs, rp = _rs_rp(tmm, cfg, lam_um, th_rad, nk=nk)
        return 0.5 * (_eps_from_r(rs) + _eps_from_r(rp)), rp
    n0, ns, nL, dL = _nk_stack(cfg, lam_um, nk=nk)
    r = tmm.r(lam_um, th_rad, n0, ns, nL, dL, pol)
    return _eps_from_r(r), r

def blackbody_spectral_radiance_um(lam_um, T):
    lam_um = np.asarray(lam_um, float)
    T = float(T)
    if T <= 0:
        raise ValueError("Temperature must be > 0 K")
    x = C2_UM_K / (lam_um * T)
    den = np.expm1(np.clip(x, 1e-12, 700.0))
    return C1_RAD_UM / (lam_um ** 5 * den)

def reflectivity(tmm, cfg, lam=None, theta=None, pol=None, nk=None, **kwargs):
    md = mm.calc(tmm, cfg, lam=lam, theta=theta, mode="R", pol=pol, nk=nk, **kwargs)
    md.mode = "Reflectivity"
    md.meta.update(dict(plot_kind="map", x_label="Wavelength (µm)", y_label="Angle (deg)", cbar_label="R", title="Reflectivity"))
    return md

def emissivity(tmm, cfg, lam=None, theta=None, pol=None, nk=None, **kwargs):
    md = mm.calc(tmm, cfg, lam=lam, theta=theta, mode="E", pol=pol, nk=nk, **kwargs)
    md.mode = "Emissivity"
    md.meta.update(dict(plot_kind="map", x_label="Wavelength (µm)", y_label="Angle (deg)", cbar_label="ε", title="Emissivity", source_assumption="opaque: ε=1-R"))
    return md

def band_averaged_emissivity(tmm, cfg, band=None, theta=None, pol=None, nk=None, lamN=None, **kwargs):
    if band is None:
        lam0, lam1 = float(cfg.get("band_lam0", 8.0)), float(cfg.get("band_lam1", 14.0))
    else:
        lam0, lam1 = map(float, band)
    if lam1 <= lam0:
        raise ValueError("band upper wavelength must be greater than lower wavelength")
    n = int(lamN if lamN is not None else cfg.get("band_lamN", cfg.get("lamN", 501)))
    lam_um = np.linspace(lam0, lam1, max(n, 2))
    th_deg = _theta_deg(cfg, theta)
    th_rad = _theta_rad(th_deg)
    pol = str(cfg.get("pol") if pol is None else pol).lower()
    eps, r = _base_eps(tmm, cfg, lam_um, th_rad, pol, nk=nk)
    val = _trapz(eps, lam_um, axis=0) / (lam1 - lam0)
    meta = _pack_meta(cfg, dict(plot_kind="line", x=th_deg, x_label="Angle (deg)", y_label="Band-averaged emissivity", cbar_label="ε_avg", title="Band-averaged emissivity", band_um=(lam0, lam1), lam_integration_points=lam_um.size, source_assumption="opaque: ε=1-R"))
    return mm.MapData(mode="Band-averaged Emissivity", pol=pol, lam_um=lam_um, theta_deg=th_deg, theta_rad=th_rad, val=val, r=r, meta=meta)

def planck_weighted_emissive_power(tmm, cfg, lam=None, theta=None, pol=None, nk=None, temperature=None, quantity=None, **kwargs):
    lam_um = _lam(cfg, lam)
    th_deg = _theta_deg(cfg, theta)
    th_rad = _theta_rad(th_deg)
    pol = str(cfg.get("pol") if pol is None else pol).lower()
    T = float(cfg.get("temperature", 300.0) if temperature is None else temperature)
    q = str(cfg.get("planck_quantity", "flux") if quantity is None else quantity).lower()
    eps, r = _base_eps(tmm, cfg, lam_um, th_rad, pol, nk=nk)
    Ib = blackbody_spectral_radiance_um(lam_um, T)[:, None]
    if q in ("radiance", "intensity", "spectral_radiance"):
        val = eps * Ib
        label = "W m^-2 sr^-1 µm^-1"
        title = "Planck-weighted spectral radiance"
    else:
        wth = 2.0 * np.pi * np.cos(th_rad) * np.sin(th_rad)
        val = eps * Ib * wth[None, :]
        label = "W m^-2 µm^-1 rad^-1"
        title = "Planck-weighted emissive-power integrand"
    meta = _pack_meta(cfg, dict(plot_kind="map", x_label="Wavelength (µm)", y_label="Angle (deg)", cbar_label=label, title=title, temperature_K=T, planck_quantity=q, source_assumption="opaque: ε=1-R"))
    return mm.MapData(mode="Planck-weighted emissive power", pol=pol, lam_um=lam_um, theta_deg=th_deg, theta_rad=th_rad, val=val, r=r, meta=meta)

def hemispherical_emissivity(tmm, cfg, lam=None, pol=None, nk=None, theta_points=None, **kwargs):
    lam_um = _lam(cfg, lam)
    ntheta = int(theta_points if theta_points is not None else cfg.get("hemi_thetaN", cfg.get("thN", 721)))
    th_deg = np.linspace(0.0, 89.999, max(ntheta, 2))
    th_rad = np.deg2rad(th_deg)
    pol = str(cfg.get("pol") if pol is None else pol).lower()
    eps, r = _base_eps(tmm, cfg, lam_um, th_rad, pol, nk=nk)
    w = np.cos(th_rad) * np.sin(th_rad)
    val = 2.0 * _trapz(eps * w[None, :], th_rad, axis=1)
    meta = _pack_meta(cfg, dict(plot_kind="line", x=lam_um, x_label="Wavelength (µm)", y_label="Hemispherical emissivity", cbar_label="ε_h", title="Hemispherical emissivity", theta_integration_deg=(0.0, 90.0), theta_integration_points=th_deg.size, source_assumption="azimuthally symmetric, opaque: ε=1-R"))
    return mm.MapData(mode="Hemispherical emissivity", pol=pol, lam_um=lam_um, theta_deg=th_deg, theta_rad=th_rad, val=val, r=r, meta=meta)

def polarization_contrast(tmm, cfg, lam=None, theta=None, metric=None, nk=None, **kwargs):
    lam_um = _lam(cfg, lam)
    th_deg = _theta_deg(cfg, theta)
    th_rad = _theta_rad(th_deg)

    rs, rp = _rs_rp(tmm, cfg, lam_um, th_rad, nk=nk)

    Rs, Rp = np.abs(rs) ** 2, np.abs(rp) ** 2
    Es, Ep = 1.0 - Rs, 1.0 - Rp

    m0 = str(cfg.get("contrast_metric", "(εₚ-εₛ)/(εₚ+εₛ)") if metric is None else metric).strip().lower()
    m = CONTRAST_ALIASES.get(m0, m0)
    tiny = 1e-300

    if m == "eps_diff":
        val = Ep - Es
        label = "εₚ - εₛ"
        title = "Polarization contrast: εₚ - εₛ"
    elif m == "r_diff":
        val = Rp - Rs
        label = "Rₚ - Rₛ"
        title = "Polarization contrast: Rₚ - Rₛ"
    elif m == "r_norm":
        val = (Rp - Rs) / (Rp + Rs + tiny)
        label = "(Rₚ - Rₛ)/(Rₚ + Rₛ)"
        title = "Normalized reflectivity contrast"
    else:
        val = (Ep - Es) / (Ep + Es + tiny)
        label = "(εₚ - εₛ)/(εₚ + εₛ)"
        title = "Normalized emissivity contrast"

    meta = _pack_meta(cfg, dict(
        plot_kind="map",
        x_label="Wavelength (µm)",
        y_label="Angle (deg)",
        cbar_label=label,
        title=title,
        contrast_metric=m,
        source_assumption="opaque: ε=1-R"
    ))

    return mm.MapData(
        mode="Polarization contrast",
        pol="s,p",
        lam_um=lam_um,
        theta_deg=th_deg,
        theta_rad=th_rad,
        val=val,
        r=rp,
        meta=meta
    )

def ellipsometry(tmm, cfg, lam=None, theta=None, output=None, nk=None, **kwargs):
    lam_um = _lam(cfg, lam)
    th_deg = _theta_deg(cfg, theta)
    th_rad = _theta_rad(th_deg)

    rs, rp = _rs_rp(tmm, cfg, lam_um, th_rad, nk=nk)

    with np.errstate(divide="ignore", invalid="ignore"):
        rho = rp / rs

    rho = np.where(np.abs(rs) > 1e-300, rho, np.nan + 1j * np.nan)

    out0 = str(cfg.get("ellip_output", "Ψ angle (deg)") if output is None else output).strip().lower()
    out = ELLIP_ALIASES.get(out0, out0)

    psi = np.arctan(np.abs(rho))
    delta = np.angle(rho)

    if out == "psi_rad":
        val = psi
        label = "Ψ (rad)"
        title = "Ellipsometry Ψ angle"
    elif out == "delta_rad":
        val = delta
        label = "Δ (rad)"
        title = "Ellipsometry Δ phase"
    elif out == "delta_deg":
        val = np.rad2deg(delta)
        label = "Δ (deg)"
        title = "Ellipsometry Δ phase"
    elif out == "rho_abs":
        val = np.abs(rho)
        label = "|ρ|"
        title = "Ellipsometry |ρ| amplitude"
    else:
        val = np.rad2deg(psi)
        label = "Ψ (deg)"
        title = "Ellipsometry Ψ angle"

    meta = _pack_meta(cfg, dict(
        plot_kind="map",
        x_label="Wavelength (µm)",
        y_label="Angle (deg)",
        cbar_label=label,
        title=title,
        ellip_output=out,
        rho=rho
    ))

    return mm.MapData(
        mode="Ellipsometry",
        pol="s,p",
        lam_um=lam_um,
        theta_deg=th_deg,
        theta_rad=th_rad,
        val=val,
        r=rho,
        meta=meta
    )

def calc(tmm, cfg, lam=None, theta=None, mode=None, pol=None, nk=None, **kwargs):
    key = _mode_key(cfg.get("mode") if mode is None else mode)
    if key == "reflectivity":
        return reflectivity(tmm, cfg, lam=lam, theta=theta, pol=pol, nk=nk, **kwargs)
    if key == "emissivity":
        return emissivity(tmm, cfg, lam=lam, theta=theta, pol=pol, nk=nk, **kwargs)
    if key == "band_avg_emissivity":
        return band_averaged_emissivity(tmm, cfg, theta=theta, pol=pol, nk=nk, **kwargs)
    if key == "planck_power":
        return planck_weighted_emissive_power(tmm, cfg, lam=lam, theta=theta, pol=pol, nk=nk, **kwargs)
    if key == "hemispherical_emissivity":
        return hemispherical_emissivity(tmm, cfg, lam=lam, pol=pol, nk=nk, **kwargs)
    if key == "polarization_contrast":
        return polarization_contrast(tmm, cfg, lam=lam, theta=theta, nk=nk, **kwargs)
    if key == "ellipsometry":
        return ellipsometry(tmm, cfg, lam=lam, theta=theta, nk=nk, **kwargs)
    raise ValueError(f"Unsupported map type: {mode}")