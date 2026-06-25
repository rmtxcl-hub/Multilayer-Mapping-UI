import os
import sqlite3
import numpy as np
import re

DB_PATH = os.path.join(os.path.dirname(__file__), "data_f.sqlite")

ADDITIONAL_TXT_FILES = {
    "paths": ["GaN-Hoffmann.txt", "GaN_strained-Hoffmann.txt", "SiC-Hoffmann.txt", "AlN-Hoffmann.txt"],
    "dir": None,
}

_VM_RE = re.compile(r"^\s*vm\s*\(\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\)\s*$")

def _vm_parse(key):
    if not isinstance(key, str): return None
    m = _VM_RE.match(key)
    if not m: return None
    return float(m.group(1)), float(m.group(2))

def _arr(x):
    if isinstance(x, str):
        s = x.strip().lower().replace(" ", "")
        if "-" in s:
            p = s.split("-")
            a, b = float(p[0]), float(p[1])
            n = int(p[2]) if len(p) >= 3 else 501
            return np.linspace(a, b, n)
        return np.array([float(s)], float)
    if np.isscalar(x):
        return np.array([float(x)], float)
    return np.asarray(x, float).ravel()

def _um_fix(w):
    return np.asarray(w, float)

def _rad(theta_deg):
    return np.deg2rad(_arr(theta_deg))

def _wl_to_um(wl):
    w = np.asarray(wl, float)
    if w.size == 0: return w
    mx = float(np.nanmax(w))
    if mx <= 0: return w
    if mx < 1e-3: return w * 1e6
    if mx > 100.0 and mx < 1e6: return w / 1000.0
    return w

def _read_txt_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if (not s) or s[0] in "#%": continue
            p = s.replace(",", " ").split()
            if len(p) < 2: continue
            try:
                if len(p) >= 3: rows.append((float(p[0]), float(p[1]), float(p[2])))
                else: rows.append((float(p[0]), float(p[1])))
            except ValueError:
                continue
    if not rows: raise ValueError(f"Empty/invalid txt: {path}")
    arr = np.array(rows, float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3): raise ValueError(f"Bad txt columns (need 2 or 3): {path}")
    i = np.argsort(arr[:, 0])
    return arr[i]

def _stem(path):
    return os.path.splitext(os.path.basename(str(path)))[0]

def _split_nk_suffix(stem):
    s = str(stem)
    lo = s.lower()
    for suf in ("-n", "_n"):
        if lo.endswith(suf): return s[: -len(suf)], "n"
    for suf in ("-k", "_k"):
        if lo.endswith(suf): return s[: -len(suf)], "k"
    return None, None

class NK:
    def __init__(self, path=DB_PATH, additional_txt_files=None):
        self.path = path
        self._c = {}
        self._txt = {}
        self._alias = {}
        self._nk_cache = {}
        cfg = additional_txt_files
        if cfg is None: cfg = ADDITIONAL_TXT_FILES
        self._load_additional_txt(cfg)

    def _load_additional_txt(self, cfg):
        if not cfg: return
        paths = []
        if isinstance(cfg, (list, tuple)):
            paths.extend([str(p) for p in cfg])
        elif isinstance(cfg, dict):
            for p in cfg.get("paths", []) or []: paths.append(str(p))
            d = cfg.get("dir", None)
            if d:
                d = str(d)
                if os.path.isdir(d):
                    for fn in os.listdir(d):
                        if fn.lower().endswith(".txt"):
                            paths.append(os.path.join(d, fn))
        else:
            raise TypeError("additional_txt_files must be dict/list/tuple")

        paths = [p for p in paths if p and os.path.isfile(p)]
        if not paths: return

        sep = {}
        for p in paths:
            st = _stem(p)
            a = _read_txt_rows(p)
            wl_um = _wl_to_um(a[:, 0])

            if a.shape[1] == 3:
                key = st
                wn = np.asarray(wl_um, float); nv = np.asarray(a[:, 1], float)
                wk = np.asarray(wl_um, float); kv = np.asarray(a[:, 2], float)
                self._txt[key] = (wn, nv, wk, kv)
                self._alias[key] = key
                continue

            base, kind = _split_nk_suffix(st)
            if base is None:
                key = st
                wn = np.asarray(wl_um, float); nv = np.asarray(a[:, 1], float)
                wk = np.array([], float); kv = np.array([], float)
                self._txt[key] = (wn, nv, wk, kv)
                self._alias[key] = key
                continue

            if base not in sep: sep[base] = {}
            sep[base][kind] = (np.asarray(wl_um, float), np.asarray(a[:, 1], float))
            self._alias[st] = base

        for base, d in sep.items():
            wn = d.get("n", (np.array([], float), np.array([], float)))[0]
            nv = d.get("n", (np.array([], float), np.array([], float)))[1]
            wk = d.get("k", (np.array([], float), np.array([], float)))[0]
            kv = d.get("k", (np.array([], float), np.array([], float)))[1]
            if wn.size == 0: continue
            self._txt[base] = (wn, nv, wk, kv)
            self._alias[base] = base

    def _resolve_key(self, key):
        vm = _vm_parse(key)
        if vm is not None:
            n, k = vm
            return f"vm({n},{k})"
        k = str(key)
        if k in self._alias: return self._alias[k]
        return k

    def _lam_cache_key(self, lam):
        a = np.ascontiguousarray(np.asarray(lam, float).ravel())
        return (a.shape, a.dtype.str, a.tobytes())

    def _interp_nk_loaded(self, wn, nv, wk, kv, lam):
        if wn.size == 0:
            raise ValueError("no n")
        n = np.interp(lam, wn, nv, left=nv[0], right=nv[-1])
        if wk.size == 0:
            k = np.zeros_like(lam, float)
        else:
            k = np.interp(lam, wk, kv, left=kv[0], right=kv[-1])
        return n + 1j * k

    def keys(self):
        out = []
        if self.path and os.path.isfile(self.path):
            with sqlite3.connect(self.path) as con:
                out.extend([r[0] for r in con.execute("SELECT dataset_key FROM meta ORDER BY dataset_key")])
        out.extend(sorted(self._alias.keys()))
        seen = set(); uniq = []
        for k in out:
            if k not in seen:
                seen.add(k); uniq.append(k)
        return uniq

    def meta(self, key):
        vm = _vm_parse(key)
        if vm is not None:
            n, k = vm
            return {"dataset_key": self._resolve_key(key), "source": "vm", "wl_min": None, "wl_max": None, "n_points": None, "k_points": None, "n_const": n, "k_const": k}

        k = self._resolve_key(key)
        if k in self._txt:
            wn, nv, wk, kv = self._txt[k]
            return {"dataset_key": k, "source": "txt", "wl_min": float(np.nanmin(wn)) if wn.size else None, "wl_max": float(np.nanmax(wn)) if wn.size else None, "n_points": int(wn.size), "k_points": int(wk.size)}

        with sqlite3.connect(self.path) as con:
            r = con.execute("SELECT * FROM meta WHERE dataset_key=?", (k,)).fetchone()
            if r is None: raise KeyError(key)
            cols = [c[1] for c in con.execute("PRAGMA table_info(meta)").fetchall()]
            return dict(zip(cols, r))

    def _load(self, key):
        k = self._resolve_key(key)
        if k in self._c: return self._c[k]
        if k in self._txt:
            self._c[k] = self._txt[k]
            return self._c[k]

        with sqlite3.connect(self.path) as con:
            nrows = con.execute("SELECT wl,n FROM n_data WHERE dataset_key=? ORDER BY wl", (k,)).fetchall()
            krows = con.execute("SELECT wl,k FROM k_data WHERE dataset_key=? ORDER BY wl", (k,)).fetchall()

        wn = np.array([x for x, _ in nrows], float) if nrows else np.array([], float)
        nv = np.array([y for _, y in nrows], float) if nrows else np.array([], float)
        wk = np.array([x for x, _ in krows], float) if krows else np.array([], float)
        kv = np.array([y for _, y in krows], float) if krows else np.array([], float)

        self._c[k] = (wn, nv, wk, kv)
        return self._c[k]

    def n(self, key, lam_um):
        lam = np.asarray(lam_um, float)
        vm = _vm_parse(key)
        if vm is not None:
            n, _ = vm
            return np.full_like(lam, n, float)
        wn, nv, _, _ = self._load(key)
        if wn.size == 0: raise ValueError(f"{key}: no n")
        return np.interp(lam, wn, nv, left=nv[0], right=nv[-1])

    def k(self, key, lam_um):
        lam = np.asarray(lam_um, float)
        vm = _vm_parse(key)
        if vm is not None:
            _, k = vm
            return np.full_like(lam, k, float)
        _, _, wk, kv = self._load(key)
        if wk.size == 0: return np.zeros_like(lam, float)
        return np.interp(lam, wk, kv, left=kv[0], right=kv[-1])

    def nk(self, key, lam_um):
        lam = np.asarray(lam_um, float)
        vm = _vm_parse(key)
        if vm is not None:
            n, k = vm
            return np.full_like(lam, n, float) + 1j * np.full_like(lam, k, float)

        k = self._resolve_key(key)
        ck = (k, self._lam_cache_key(lam))
        if ck in self._nk_cache:
            return self._nk_cache[ck]

        wn, nv, wk, kv = self._load(k)
        out = self._interp_nk_loaded(wn, nv, wk, kv, lam)
        self._nk_cache[ck] = out
        return out

    def prep(self, lam, theta, pol, dL, film_keys, sub_key, cover_n=1.0, cover_k=0.0):
        lam_um = _um_fix(_arr(lam))
        th_rad = _rad(theta)
        pol = str(pol).lower()
        if pol not in ("s", "p"): raise ValueError("pol must be 's' or 'p'")
        film_keys = list(film_keys)
        dL = list(map(float, dL))
        if len(dL) != len(film_keys): raise ValueError("len(dL)!=len(film_keys)")
        n0 = complex(cover_n, cover_k)
        ns = self.nk(sub_key, lam_um)
        nL = [self.nk(k, lam_um) for k in film_keys]
        return dict(lam_um=lam_um, theta_rad=th_rad, pol=pol, n0=n0, ns=ns, nL=nL, dL=dL, film_keys=film_keys, sub_key=sub_key, cover_n=float(cover_n), cover_k=float(cover_k))