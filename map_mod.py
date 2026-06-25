import numpy as np
from dataclasses import dataclass

def _grid(x, N=501):
    if isinstance(x, str):
        s=x.strip().lower().replace(" ","")
        if "-" in s:
            p=s.split("-"); a,b=float(p[0]),float(p[1]); n=int(p[2]) if len(p)>=3 else N
            return np.linspace(a,b,n)
        return np.array([float(s)], float)
    if np.isscalar(x): return np.array([float(x)], float)
    return np.asarray(x, float).ravel()

def _um_fix(w): return np.asarray(w, float)
def _th_deg(x): return _grid(x)
def _th_rad(th_deg): return np.deg2rad(np.asarray(th_deg, float))

@dataclass
class MapData:
    mode:str; pol:str
    lam_um:np.ndarray; theta_deg:np.ndarray; theta_rad:np.ndarray
    val:np.ndarray; r:np.ndarray; meta:dict
    def pack(self):
        return dict(mode=self.mode, pol=self.pol, lam_um=self.lam_um, theta_deg=self.theta_deg,
                    theta_rad=self.theta_rad, val=self.val, r=self.r, meta=self.meta)

def _cpu_eval(tmm, lam_um, th_rad, n0, ns, nL, dL, pol, mode):
    if pol == "both":
        r_s = tmm.r(lam_um, th_rad, n0, ns, nL, dL, "s")
        r_p = tmm.r(lam_um, th_rad, n0, ns, nL, dL, "p")
        R = 0.5 * (np.abs(r_s) ** 2 + np.abs(r_p) ** 2)
        r = r_p
    else:
        r = tmm.r(lam_um, th_rad, n0, ns, nL, dL, pol)
        R = np.abs(r) ** 2
    val = R if mode == "R" else (1.0 - R)
    return r, val

def calc(tmm, cfg, lam=None, theta=None, mode="R", pol=None, nk=None,
         use_gpu=False, gpu_theta_batch=32, gpu_free=True, gpu_device=0):
    lam_um = _um_fix(_grid(cfg["lam_um"] if lam is None else lam))
    th_deg = _th_deg(cfg["theta_rad"] * 180 / np.pi if theta is None else theta)
    th_rad = _th_rad(th_deg)
    pol = (cfg["pol"] if pol is None else pol).lower()
    if pol not in ("s", "p", "both"):
        raise ValueError("pol must be 's', 'p', or 'both'")
    mode = str(mode).upper()
    if mode not in ("R", "E"):
        raise ValueError("mode must be 'R' or 'E'")

    n0, dL = cfg["n0"], cfg["dL"]
    if nk is not None:
        ns = nk.nk(cfg["sub_key"], lam_um)
        nL = [nk.nk(k, lam_um) for k in cfg["film_keys"]]
    else:
        ns, nL = cfg["ns"], cfg["nL"]
        if np.asarray(ns).shape[0] != lam_um.size:
            raise ValueError("cfg nk grid != map lam; pass nk=NK() or match lam grids")

    if not use_gpu:
        r, val = _cpu_eval(tmm, lam_um, th_rad, n0, ns, nL, dL, pol, mode)
        meta = dict(
            n0=n0, film_keys=cfg.get("film_keys"), sub_key=cfg.get("sub_key"),
            dL=list(dL), cover_n=cfg.get("cover_n"), cover_k=cfg.get("cover_k")
        )
        return MapData(mode=mode, pol=pol, lam_um=lam_um, theta_deg=th_deg, theta_rad=th_rad, val=val, r=r, meta=meta)

    try:
        import cupy as cp
    except Exception:
        r, val = _cpu_eval(tmm, lam_um, th_rad, n0, ns, nL, dL, pol, mode)
        meta = dict(
            n0=n0, film_keys=cfg.get("film_keys"), sub_key=cfg.get("sub_key"),
            dL=list(dL), cover_n=cfg.get("cover_n"), cover_k=cfg.get("cover_k")
        )
        return MapData(mode=mode, pol=pol, lam_um=lam_um, theta_deg=th_deg, theta_rad=th_rad, val=val, r=r, meta=meta)

    cp.cuda.Device(int(gpu_device)).use()
    xp = cp
    lamg = xp.asarray(lam_um, float)
    nsg = xp.asarray(ns, complex)
    nLg = [xp.asarray(a, complex) for a in nL]
    B = max(1, int(gpu_theta_batch))

    if pol == "both":
        rGs = xp.empty((lam_um.size, th_rad.size), dtype=complex)
        rGp = xp.empty((lam_um.size, th_rad.size), dtype=complex)
        for j0 in range(0, th_rad.size, B):
            j1 = min(th_rad.size, j0 + B)
            thb = xp.asarray(th_rad[j0:j1], float)
            rGs[:, j0:j1] = tmm.r(lamg, thb, n0, nsg, nLg, dL, "s")
            rGp[:, j0:j1] = tmm.r(lamg, thb, n0, nsg, nLg, dL, "p")
        RG = 0.5 * (xp.abs(rGs) ** 2 + xp.abs(rGp) ** 2)
        rG = rGp
    else:
        rG = xp.empty((lam_um.size, th_rad.size), dtype=complex)
        for j0 in range(0, th_rad.size, B):
            j1 = min(th_rad.size, j0 + B)
            thb = xp.asarray(th_rad[j0:j1], float)
            rG[:, j0:j1] = tmm.r(lamg, thb, n0, nsg, nLg, dL, pol)
        RG = xp.abs(rG) ** 2

    valG = RG if mode == "R" else (1.0 - RG)
    r = xp.asnumpy(rG)
    val = xp.asnumpy(valG)

    if gpu_free:
        try:
            xp.get_default_memory_pool().free_all_blocks()
            xp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    meta = dict(
        n0=n0, film_keys=cfg.get("film_keys"), sub_key=cfg.get("sub_key"),
        dL=list(dL), cover_n=cfg.get("cover_n"), cover_k=cfg.get("cover_k")
    )
    return MapData(mode=mode, pol=pol, lam_um=lam_um, theta_deg=th_deg, theta_rad=th_rad, val=val, r=r, meta=meta)

def _fmt_list(x):
    if x is None: return "(none)"
    if isinstance(x,(list,tuple,np.ndarray)):
        if len(x)==0: return "(none)"
        return ", ".join(str(v) for v in x)
    return str(x)

def _fmt_thickness_um(dL):
    if dL is None: return "(none)"
    try: a=[float(v) for v in dL]
    except Exception: return _fmt_list(dL)
    return ", ".join(f"{v:.6g}" for v in a)

def show(md, clip=(0.5,99.5), vlim=None, cmap="turbo", ax=None):
    import matplotlib.pyplot as plt
    W,A,Z=md.lam_um,md.theta_deg,md.val
    vmin,vmax=(0.0,1.0) if vlim is None else (float(vlim[0]),float(vlim[1]))
    if ax is None: fig,ax=plt.subplots(figsize=(7.5,5.5))
    else: fig=ax.figure
    im=ax.pcolormesh(W,A,Z.T,shading="auto",vmin=vmin,vmax=vmax,cmap=cmap)
    ax.set_xlabel("Wavelength (um)"); ax.set_ylabel("Angle (deg)")
    ttl0="Reflectivity" if md.mode=="R" else "Emissivity"
    sub=(md.meta or {}).get("sub_key"); films=(md.meta or {}).get("film_keys"); dL=(md.meta or {}).get("dL")
    ax.set_title(f"{ttl0} map ({md.pol}-pol) | sub={sub}\nfilms(top->bot): {_fmt_list(films)}\nd_um(top->bot): {_fmt_thickness_um(dL)}",
                 fontsize=9,pad=10)
    fig.colorbar(im,ax=ax,label=("R" if md.mode=="R" else "eps"))
    fig.tight_layout()
    return ax