import numpy as np
import map_mod as mm
import field_collection as fc

TINY = 1e-300

FIELD_MAP_TYPES = [
    "E-field intensity map",
    "B-field intensity map",
    "Angular field-depth map",
    "Wavelength field-depth map",
]

FIELD_COMPONENTS = ["total", "x", "y", "z", "tangential", "normal"]
FIELD_NORMALIZATIONS = ["none", "incident", "maximum", "log incident"]
FIELD_SLICES = ["lambda-theta", "lambda-depth", "theta-depth"]
FIELD_DEPTH_REGIONS = ["stack", "selected layer"]

MAP_ALIASES = {
    "e-field intensity map": "field_intensity",
    "electric field intensity map": "field_intensity",
    "e field intensity map": "field_intensity",
    "electric field intensity": "field_intensity",
    "b-field intensity map": "field_intensity",
    "magnetic field intensity map": "field_intensity",
    "b field intensity map": "field_intensity",
    "magnetic field intensity": "field_intensity",
    "field intensity": "field_intensity",
    "field intensity map": "field_intensity",
    "angular field-depth map": "angular_field_depth",
    "angular field depth map": "angular_field_depth",
    "theta-depth field map": "angular_field_depth",
    "theta depth field map": "angular_field_depth",
    "wavelength field-depth map": "wavelength_field_depth",
    "wavelength field depth map": "wavelength_field_depth",
    "lambda-depth field map": "wavelength_field_depth",
    "lambda depth field map": "wavelength_field_depth",
}

def _mode_key(mode):
    s = str(mode or "").strip().lower()
    return MAP_ALIASES.get(s, s.replace(" ", "_").replace("-", "_"))

def _grid(x, N=501):
    return mm._grid(x, N=N)

def _lam(cfg, lam=None):
    return mm._um_fix(_grid(cfg["lam_um"] if lam is None else lam, int(cfg.get("lamN", 501))))

def _theta_deg(cfg, theta=None):
    src = cfg["theta_rad"] * 180.0 / np.pi if theta is None else theta
    return mm._th_deg(src)

def _theta_rad(th_deg):
    return mm._th_rad(th_deg)

def _first_present(cfg, keys, default=None):
    for k in keys:
        if k in cfg and cfg[k] is not None and str(cfg[k]).strip() != "":
            return cfg[k]
    return default

def _field_kind(mode, cfg):
    v = _first_present(cfg, ("field_kind", "field", "field_quantity"), None)
    if v is None:
        m = str(mode or cfg.get("mode", "")).lower()
        v = "B" if "b-field" in m or "magnetic" in m else "E"
    s = str(v).strip().upper()
    if s in ("E", "E-FIELD", "ELECTRIC", "ELECTRIC FIELD"):
        return "E"
    if s in ("B", "B-FIELD", "MAGNETIC", "MAGNETIC FIELD", "H", "H-FIELD"):
        return "B"
    raise ValueError("field_kind must be 'E' or 'B'")

def _component(cfg):
    s = str(_first_present(cfg, ("field_component", "component"), "total")).strip().lower()
    aliases = {"aggregate": "total", "all": "total", "norm": "total", "magnitude": "total", "mag": "total", "xy": "tangential", "xz": "tangential"}
    s = aliases.get(s, s)
    if s not in FIELD_COMPONENTS:
        raise ValueError("field_component must be total, x, y, z, tangential, or normal")
    return s

def _normalization(cfg):
    s = str(_first_present(cfg, ("field_normalization", "normalization", "field_norm"), "incident")).strip().lower()
    aliases = {
        "raw": "none",
        "no": "none",
        "off": "none",
        "incident-normalized": "incident",
        "incident normalized": "incident",
        "inc": "incident",
        "max": "maximum",
        "map max": "maximum",
        "map maximum": "maximum",
        "maximum-normalized": "maximum",
        "normalized 0-1": "maximum",
        "log": "log incident",
        "log10": "log incident",
        "log10 incident": "log incident",
        "log incident-normalized": "log incident",
    }
    s = aliases.get(s, s)
    if s not in FIELD_NORMALIZATIONS:
        raise ValueError("field_normalization must be none, incident, maximum, or log incident")
    return s

def _slice_type(mode_key, cfg):
    if mode_key == "angular_field_depth":
        return "theta-depth"
    if mode_key == "wavelength_field_depth":
        return "lambda-depth"

    s = str(_first_present(cfg, ("field_slice", "slice_type", "field_view"), "lambda-theta")).strip().lower()
    aliases = {
        "λ-θ": "lambda-theta",
        "lambda-theta": "lambda-theta",
        "lam-theta": "lambda-theta",
        "wavelength-angle": "lambda-theta",
        "fix z": "lambda-theta",
        "fixed z": "lambda-theta",
        "λ-z": "lambda-depth",
        "lambda-z": "lambda-depth",
        "lam-z": "lambda-depth",
        "wavelength-depth": "lambda-depth",
        "fix theta": "lambda-depth",
        "fixed theta": "lambda-depth",
        "θ-z": "theta-depth",
        "theta-z": "theta-depth",
        "angle-depth": "theta-depth",
        "angular-depth": "theta-depth",
        "fix lambda": "theta-depth",
        "fixed lambda": "theta-depth",
    }
    s = aliases.get(s, s)
    if s not in FIELD_SLICES:
        raise ValueError("field_slice must be lambda-theta, lambda-depth, or theta-depth")
    return s

def _pol_list(pol):
    p = str(pol).strip().lower()
    if p == "both":
        return ["s", "p"]
    if p in ("s", "p"):
        return [p]
    raise ValueError("pol must be 's', 'p', or 'both'")

def _clamp_layer(layer, dL):
    if not dL:
        raise ValueError("At least one film layer is required for field maps")
    j = int(layer) - 1
    if j < 0 or j >= len(dL):
        raise ValueError("field_layer must be a 1-based layer index inside the film stack")
    return j

def _depth_region(cfg):
    s = str(_first_present(cfg, ("field_depth_region", "depth_region"), "stack")).strip().lower()
    aliases = {"full": "stack", "full stack": "stack", "film": "stack", "film stack": "stack", "all": "stack", "layer": "selected layer", "selected": "selected layer"}
    s = aliases.get(s, s)
    if s not in FIELD_DEPTH_REGIONS:
        raise ValueError("field_depth_region must be 'stack' or 'selected layer'")
    return s

def _points_per_layer(cfg):
    n = int(float(_first_present(cfg, ("field_points_per_layer", "points_per_layer", "z_points_per_layer"), 80)))
    return max(n, 2)

def _z_points(cfg):
    n = int(float(_first_present(cfg, ("field_z_points", "z_points", "zN", "field_zN"), 200)))
    return max(n, 2)

def _layer_from_abs_z(z_abs, dL):
    z_abs = float(z_abs)
    if z_abs < 0 or z_abs > sum(dL):
        raise ValueError("field_z_abs_um must lie inside the film stack")
    edges = np.r_[0.0, np.cumsum(dL)]
    if z_abs == edges[-1]:
        return len(dL) - 1, float(dL[-1])
    j = int(np.searchsorted(edges, z_abs, side="right") - 1)
    j = max(0, min(j, len(dL) - 1))
    return j, z_abs - edges[j]

def _fixed_z(cfg, dL):
    if "field_z_abs_um" in cfg and cfg["field_z_abs_um"] is not None and str(cfg["field_z_abs_um"]).strip() != "":
        return _layer_from_abs_z(float(cfg["field_z_abs_um"]), dL)

    j = _clamp_layer(_first_present(cfg, ("field_layer", "selected_layer", "layer"), 1), dL)

    if "field_z_um" in cfg and cfg["field_z_um"] is not None and str(cfg["field_z_um"]).strip() != "":
        z = float(cfg["field_z_um"])
    else:
        f = float(_first_present(cfg, ("field_z_frac", "z_frac", "layer_fraction"), 0.5))
        z = f * float(dL[j])

    if z < 0 or z > dL[j]:
        raise ValueError("field_z_um or field_z_frac places the slice outside the selected layer")

    return j, z

def _depth_samples(cfg, dL):
    region = _depth_region(cfg)
    edges = np.r_[0.0, np.cumsum(dL)]

    if region == "selected layer":
        j = _clamp_layer(_first_present(cfg, ("field_layer", "selected_layer", "layer"), 1), dL)
        n = _z_points(cfg)
        zloc = np.linspace(0.0, float(dL[j]), n)
        zabs = edges[j] + zloc
        layers = np.full(n, j, int)
        return zabs, layers, zloc

    p = _points_per_layer(cfg)
    z_abs, layers, z_loc = [], [], []
    for j, dj in enumerate(dL):
        zl = np.linspace(0.0, float(dj), p, endpoint=False)
        z_abs.extend(edges[j] + zl)
        z_loc.extend(zl)
        layers.extend([j] * zl.size)

    z_abs.append(edges[-1])
    z_loc.append(float(dL[-1]))
    layers.append(len(dL) - 1)

    return np.asarray(z_abs, float), np.asarray(layers, int), np.asarray(z_loc, float)

def _component_scalar(comps, field, component):
    names = ("Ex", "Ey", "Ez") if field == "E" else ("Hx", "Hy", "Hz")
    x, y, z = (comps[n] for n in names)

    if component == "x":
        return np.abs(x) ** 2
    if component == "y":
        return np.abs(y) ** 2
    if component in ("z", "normal"):
        return np.abs(z) ** 2
    if component == "tangential":
        return np.abs(x) ** 2 + np.abs(y) ** 2
    return np.abs(x) ** 2 + np.abs(y) ** 2 + np.abs(z) ** 2

def _components_from_amp(tmm, amp, nL, pol, layer, z_um):
    j = int(layer)
    nw, nt = amp["lam_um"].size, amp["theta_deg"].size
    out = {k: np.empty((nw, nt), complex) for k in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}

    for it in range(nt):
        c = fc._components_from_amplitudes(
            amp["lam_um"],
            float(z_um),
            amp["E_plus"][j, :, it],
            amp["E_minus"][j, :, it],
            amp["q"][j, :, it],
            amp["cos_theta"][j, :, it],
            amp["sin_theta"][j, :, it],
            nL[j],
            pol,
            getattr(tmm, "clip_im_delta", 60.0),
        )
        for k in out:
            out[k][:, it] = c[k]

    out["q0"] = amp["q0"]
    return out

def _incident_normalize(v, q0, field):
    if field == "E":
        return v
    return v / np.maximum(np.abs(q0) ** 2, TINY)

def _apply_post_norm(v, normalization):
    if normalization == "maximum":
        a = v[np.isfinite(v)]
        m = float(np.nanmax(a)) if a.size else 0.0
        return v / m if m > 0 else v
    if normalization == "log incident":
        return np.log10(np.maximum(v, TINY))
    return v

def _field_volume(tmm, cfg, lam_um, theta_deg, z_layers, z_local, field, component, normalization, pol, nk):
    n0, ns, nL, dL = fc._nk_stack(cfg, lam_um, nk=nk)
    vals = []

    for pp in _pol_list(pol):
        amp = fc.amplitudes(tmm, cfg, lam=lam_um, theta=theta_deg, pol=pp, nk=nk)
        vp = np.empty((lam_um.size, theta_deg.size, len(z_local)), float)

        for iz, (j, z) in enumerate(zip(z_layers, z_local)):
            comps = _components_from_amp(tmm, amp, nL, pp, int(j), float(z))
            raw = _component_scalar(comps, field, component)
            if normalization in ("incident", "log incident"):
                raw = _incident_normalize(raw, comps["q0"], field)
            vp[:, :, iz] = raw

        vals.append(vp)

    val = vals[0] if len(vals) == 1 else 0.5 * (vals[0] + vals[1])
    val = _apply_post_norm(val, normalization)

    return val

def _cbar_label(field, component, normalization):
    symbol = "E" if field == "E" else "B"
    comp = {
        "total": f"|{symbol}|²",
        "x": f"|{symbol}x|²",
        "y": f"|{symbol}y|²",
        "z": f"|{symbol}z|²",
        "normal": f"|{symbol}z|²",
        "tangential": f"|{symbol}t|²",
    }[component]

    if normalization == "incident":
        return f"{comp} / |{symbol}inc|²"
    if normalization == "maximum":
        return f"{comp} / max({comp})"
    if normalization == "log incident":
        return f"log10({comp} / |{symbol}inc|²)"
    return f"{comp} (a.u.)"

def _title(field, component, slice_type):
    symbol = "E-field" if field == "E" else "B-field"
    names = {
        "lambda-theta": "intensity map at fixed depth",
        "lambda-depth": "wavelength field-depth map",
        "theta-depth": "angular field-depth map",
    }
    return f"{symbol} {component} {names[slice_type]}"

def _pack_meta(cfg, field, component, normalization, slice_type, extra=None):
    meta = dict(
        plot_kind="map",
        n0=cfg.get("n0"),
        film_keys=cfg.get("film_keys"),
        sub_key=cfg.get("sub_key"),
        dL=list(cfg.get("dL", [])),
        cover_n=cfg.get("cover_n"),
        cover_k=cfg.get("cover_k"),
        field_kind=field,
        field_component=component,
        field_normalization=normalization,
        field_slice=slice_type,
        cbar_label=_cbar_label(field, component, normalization),
        title=_title(field, component, slice_type),
        source_assumption="TMM internal-field reconstruction; B-field uses the nonmagnetic normalized H-field convention",
    )
    if extra:
        meta.update(extra)
    return meta

def _fixed_theta(cfg):
    return float(_first_present(cfg, ("field_fixed_theta_deg", "fixed_theta_deg", "theta_fixed_deg"), 0.0))

def _fixed_lam(cfg):
    v = _first_present(cfg, ("field_fixed_lam_um", "fixed_lam_um", "lambda_fixed_um"), None)
    if v is not None:
        return float(v)
    lam = _lam(cfg)
    return float(lam[len(lam) // 2])

def lambda_theta_map(tmm, cfg, field="E", component="total", normalization="incident", lam=None, theta=None, pol=None, nk=None):
    lam_um = _lam(cfg, lam)
    th_deg = _theta_deg(cfg, theta)
    th_rad = _theta_rad(th_deg)
    dL = list(map(float, cfg["dL"]))
    layer, z_um = _fixed_z(cfg, dL)
    val3 = _field_volume(tmm, cfg, lam_um, th_deg, [layer], [z_um], field, component, normalization, pol or cfg.get("pol", "p"), nk)
    val = val3[:, :, 0]

    r = np.full_like(val, np.nan, dtype=complex)
    meta = _pack_meta(cfg, field, component, normalization, "lambda-theta", dict(
        x=lam_um,
        y=th_deg,
        x_label="Wavelength (µm)",
        y_label="Angle (deg)",
        fixed_layer=layer + 1,
        fixed_z_um=z_um,
    ))

    return mm.MapData(
        mode=f"{field}-field intensity map",
        pol=str(pol or cfg.get("pol", "p")).lower(),
        lam_um=lam_um,
        theta_deg=th_deg,
        theta_rad=th_rad,
        val=val,
        r=r,
        meta=meta,
    )

def wavelength_field_depth_map(tmm, cfg, field="E", component="total", normalization="incident", lam=None, pol=None, nk=None):
    lam_um = _lam(cfg, lam)
    th = _fixed_theta(cfg)
    th_deg = np.array([th], float)
    z_abs, z_layers, z_local = _depth_samples(cfg, list(map(float, cfg["dL"])))
    val3 = _field_volume(tmm, cfg, lam_um, th_deg, z_layers, z_local, field, component, normalization, pol or cfg.get("pol", "p"), nk)
    val = val3[:, 0, :]

    meta = _pack_meta(cfg, field, component, normalization, "lambda-depth", dict(
        x=lam_um,
        y=z_abs,
        x_label="Wavelength (µm)",
        y_label="Depth in stack (µm)",
        fixed_theta_deg=th,
        depth_region=_depth_region(cfg),
    ))

    return mm.MapData(
        mode="Wavelength field-depth map",
        pol=str(pol or cfg.get("pol", "p")).lower(),
        lam_um=lam_um,
        theta_deg=z_abs,
        theta_rad=np.deg2rad(th_deg),
        val=val,
        r=np.full((lam_um.size, 1), np.nan + 1j * np.nan),
        meta=meta,
    )

def angular_field_depth_map(tmm, cfg, field="E", component="total", normalization="incident", theta=None, pol=None, nk=None):
    lam0 = _fixed_lam(cfg)
    lam_um = np.array([lam0], float)
    th_deg = _theta_deg(cfg, theta)
    z_abs, z_layers, z_local = _depth_samples(cfg, list(map(float, cfg["dL"])))
    val3 = _field_volume(tmm, cfg, lam_um, th_deg, z_layers, z_local, field, component, normalization, pol or cfg.get("pol", "p"), nk)
    val = val3[0, :, :]

    meta = _pack_meta(cfg, field, component, normalization, "theta-depth", dict(
        x=th_deg,
        y=z_abs,
        x_label="Angle (deg)",
        y_label="Depth in stack (µm)",
        fixed_lam_um=lam0,
        depth_region=_depth_region(cfg),
    ))

    return mm.MapData(
        mode="Angular field-depth map",
        pol=str(pol or cfg.get("pol", "p")).lower(),
        lam_um=th_deg,
        theta_deg=z_abs,
        theta_rad=np.deg2rad(th_deg),
        val=val,
        r=np.full((1, th_deg.size), np.nan + 1j * np.nan),
        meta=meta,
    )

def calc(tmm, cfg, lam=None, theta=None, mode=None, pol=None, nk=None, **kwargs):
    mode0 = cfg.get("mode") if mode is None else mode
    key = _mode_key(mode0)
    field = _field_kind(mode0, cfg)
    component = _component(cfg)
    normalization = _normalization(cfg)
    slice_type = _slice_type(key, cfg)
    pol = str(cfg.get("pol") if pol is None else pol).lower()

    if key not in ("field_intensity", "angular_field_depth", "wavelength_field_depth"):
        raise ValueError(f"Unsupported map_collection2 field map type: {mode0}")

    if slice_type == "lambda-theta":
        return lambda_theta_map(tmm, cfg, field=field, component=component, normalization=normalization, lam=lam, theta=theta, pol=pol, nk=nk)

    if slice_type == "lambda-depth":
        return wavelength_field_depth_map(tmm, cfg, field=field, component=component, normalization=normalization, lam=lam, pol=pol, nk=nk)

    if slice_type == "theta-depth":
        return angular_field_depth_map(tmm, cfg, field=field, component=component, normalization=normalization, theta=theta, pol=pol, nk=nk)

    raise ValueError(f"Unsupported field_slice: {slice_type}")
