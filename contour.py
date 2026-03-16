import numpy as np
from scipy.ndimage import (
    label, generate_binary_structure, gaussian_filter,
    binary_fill_holes, binary_closing
)

def _as_axes(M, lam, theta):
    lam=np.asarray(lam,float).ravel()
    th=np.asarray(theta,float).ravel()
    Z=np.asarray(M,float)
    if Z.shape==(th.size,lam.size): return Z,lam,th
    if Z.shape==(lam.size,th.size): return Z.T,lam,th
    raise ValueError("shape mismatch")

def _pct(X,p):
    X=np.where(np.isfinite(X),X,np.nan)
    return float(np.nanpercentile(X,float(p)))

def _otsu(x, bins=256, return_eta=False):
    x=np.asarray(x,float)
    x=x[np.isfinite(x)]
    if x.size<2:
        if return_eta: return np.nan, 0.0
        return np.nan
    x0,x1=float(x.min()),float(x.max())
    if not (x1>x0):
        if return_eta: return x0, 1.0
        return x0
    h,e=np.histogram(x,bins=int(bins),range=(x0,x1))
    h=h.astype(float)
    if h.sum()<=0:
        if return_eta: return np.nan, 0.0
        return np.nan
    p=h/h.sum()
    w=np.cumsum(p)
    c=(e[:-1]+e[1:])*0.5
    m=np.cumsum(p*c)
    mt=m[-1]
    den=w*(1.0-w)
    s=np.zeros_like(den)
    ok=den>0
    s[ok]=((mt*w[ok]-m[ok])**2)/den[ok]
    k=int(np.argmax(s[:-1]))
    t=float(c[k])
    if not return_eta: return t
    sig_b2=float(s[k])
    sig_t2=float(np.var(x,ddof=0))
    eta=0.0 if not (sig_t2>0.0 and np.isfinite(sig_b2)) else sig_b2/sig_t2
    eta=float(np.clip(eta,0.0,1.0))
    return t, eta

def _pct_rank(x, t):
    x=np.asarray(x,float).ravel()
    m=np.isfinite(x)
    if not m.any():
        return float("nan"), float("nan")
    xm=x[m]
    ple=100.0*float(np.mean(xm<=t))
    keep=100.0*float(np.mean(xm>=t))
    return ple, keep

def mask(M, lam, theta, *, mode="E", peak=None,
         thr=None, thr_pct=97.0,
         thr_hi="percentile", thr_hi_alpha=0.7, thr_hi_base="median",
         thr_lo=None, thr_lo_pct="auto",
         sig=1.2, close=2, single="largest",
         return_report=False):
    Z,lam,th=_as_axes(M,lam,theta)
    Z=np.where(np.isfinite(Z),Z,np.nan)
    if sig and sig>0: Z=gaussian_filter(Z,float(sig),mode="nearest")

    if peak is None:
        peak="high" if str(mode).upper() in ("E","A") else "low"
    V=-Z if peak=="low" else Z

    thr_hi=str(thr_hi).lower() if thr_hi is not None else "percentile"
    use_adaptive=(thr is None) and (thr_hi in ("adaptive","perblob","per_blob","auto"))

    if thr is not None:
        thi=(-thr) if peak=="low" else float(thr)
        thi_method="absolute"
    else:
        if use_adaptive:
            thi=_pct(V,97.0)
            thi_method="adaptive_per_blob"
        else:
            thi=_pct(V,thr_pct)
            thi_method="percentile"

    eta=None
    tlo_raw=None
    if thr_lo is not None:
        tlo=(-thr_lo) if peak=="low" else float(thr_lo)
        tlo_method="absolute"
    else:
        if isinstance(thr_lo_pct,str) and thr_lo_pct.lower()=="auto":
            tlo_raw,eta=_otsu(V,return_eta=True)
            if not np.isfinite(tlo_raw):
                tlo_raw=_pct(V,50.0)
                eta=1.0
            w=float(eta)
            tlo=w*float(tlo_raw) + (1.0-w)*float(thi)
            tlo_method="otsu_tight"
        else:
            p=float(thr_lo_pct) if thr_lo_pct is not None else float(thr_pct)
            tlo=_pct(V,p)
            tlo_method="percentile"

    tlo=min(float(tlo),float(thi))
    tlo_ple,tlo_keep=_pct_rank(V,tlo)
    thi_out=-float(thi) if peak=="low" else float(thi)
    tlo_out=-float(tlo) if peak=="low" else float(tlo)
    tlo_raw_out=None if tlo_raw is None else (-float(tlo_raw) if peak=="low" else float(tlo_raw))

    report=dict(
        peak=str(peak),
        mode=str(mode),
        thr_used=float(thi_out),
        thr_method=str(thi_method),
        thr_pct=float(thr_pct) if thr is None else None,
        thr_hi=str(thr_hi),
        thr_hi_alpha=(float(thr_hi_alpha) if use_adaptive else None),
        thr_hi_base=(str(thr_hi_base) if use_adaptive else None),
        thr_lo_used=float(tlo_out),
        thr_lo_method=str(tlo_method),
        thr_lo_pct=("auto" if (thr_lo is None and isinstance(thr_lo_pct,str) and thr_lo_pct.lower()=="auto") else (float(thr_lo_pct) if (thr_lo is None and thr_lo_pct is not None and not isinstance(thr_lo_pct,str)) else None)),
        thr_lo_percentile=float(tlo_ple),
        thr_lo_keep_pct=float(tlo_keep),
        thr_lo_raw=(None if tlo_raw_out is None else float(tlo_raw_out)),
        thr_lo_eta=(None if eta is None else float(eta))
    )

    lo=(V>=tlo)

    st=generate_binary_structure(2,2)
    lab,n=label(lo.astype(np.uint8),structure=st)
    if n==0:
        out=(lam,th,np.zeros_like(lo,bool),thi_out)
        return (*out,report) if return_report else out

    if use_adaptive:
        hi=np.zeros_like(lo,bool)
        a=float(np.clip(thr_hi_alpha,0.0,1.0))
        base=str(thr_hi_base).lower()
        thi_list=[]
        for i in range(1,n+1):
            ci=(lab==i)
            xi=V[ci]
            xi=xi[np.isfinite(xi)]
            if xi.size==0: continue
            mi=float(np.max(xi))
            if base in ("p25","q25","25"):
                bi=float(np.percentile(xi,25.0))
            elif base in ("p10","q10","10"):
                bi=float(np.percentile(xi,10.0))
            elif base in ("mean","avg"):
                bi=float(np.mean(xi))
            else:
                bi=float(np.median(xi))
            thi_i=bi + a*(mi-bi)
            thi_list.append(thi_i)
            hi[ci]=(V[ci]>=thi_i)
        if thi_list:
            report["thr_hi_perblob_min"]=float(np.min(thi_list))
            report["thr_hi_perblob_med"]=float(np.median(thi_list))
            report["thr_hi_perblob_max"]=float(np.max(thi_list))
        else:
            report["thr_hi_perblob_min"]=None
            report["thr_hi_perblob_med"]=None
            report["thr_hi_perblob_max"]=None
    else:
        hi=(V>=thi)

    ids=np.unique(lab[hi]); ids=ids[ids>0]
    if ids.size==0:
        out=(lam,th,np.zeros_like(lo,bool),thi_out)
        return (*out,report) if return_report else out

    m=np.isin(lab,ids)
    m=binary_fill_holes(m)
    if close and close>0:
        m=binary_closing(m,structure=st,iterations=int(close))

    if single=="largest":
        lab2,n2=label(m.astype(np.uint8),structure=st)
        if n2>1:
            cnt=np.bincount(lab2.ravel()); cnt[0]=0
            m=(lab2==int(cnt.argmax()))

    out=(lam,th,m,thi_out)
    return (*out,report) if return_report else out

def _pad_mask(M,pad):
    p=int(pad) if pad else 0
    return np.pad(M,((p,p),(p,p)),mode="constant",constant_values=False) if p>0 else M

def _pad_axes(lam,th,pad):
    p=int(pad) if pad else 0
    if p<=0: return lam,th
    dlam=float(np.median(np.diff(lam))) if lam.size>1 else 1.0
    dth=float(np.median(np.diff(th))) if th.size>1 else 1.0
    lam2=np.concatenate([lam[0]-dlam*np.arange(p,0,-1), lam, lam[-1]+dlam*np.arange(1,p+1)])
    th2=np.concatenate([th[0]-dth*np.arange(p,0,-1),  th,  th[-1]+dth*np.arange(1,p+1)])
    return lam2,th2

def segments(ax, lam, theta, mask, *, level=0.5, pad=0):
    M,lam,th=_as_axes(mask,lam,theta)
    B=_pad_mask(M.astype(bool),pad); lam2,th2=_pad_axes(lam,th,pad)
    cs=ax.contour(lam2,th2,B.astype(float),levels=[float(level)],linewidths=0.0,alpha=0.0)
    segs=cs.allsegs[0] if cs.allsegs else []
    try:
        for c in cs.collections: c.remove()
    except Exception:
        pass
    return [s for s in segs if getattr(s,"shape",(0,0))[0]>=2]

def draw(ax, lam, theta, mask, *, lw=2.0, alpha=0.9, color=None, zorder=None, pick="longest", level=0.5, pad=0):
    segs=segments(ax,lam,theta,mask,level=level,pad=pad)
    if not segs: return [],segs
    keep=segs if pick=="all" else [max(segs,key=lambda s:s.shape[0])]
    out=[]
    for s in keep:
        (ln,)=ax.plot(s[:,0],s[:,1],lw=float(lw),alpha=float(alpha),color=color,zorder=zorder)
        out.append(ln)
    return out,segs

def apply(ax, md=None, *, val=None, lam=None, theta=None, mode=None, mask_cfg=None, draw_cfg=None, return_report=False):
    if md is not None:
        val=md.val; lam=md.lam_um; theta=md.theta_deg; mode=md.mode
    if val is None or lam is None or theta is None: raise ValueError("need md or (val,lam,theta)")
    mask_cfg={} if mask_cfg is None else dict(mask_cfg)
    draw_cfg={} if draw_cfg is None else dict(draw_cfg)
    if return_report and "return_report" not in mask_cfg:
        mask_cfg["return_report"]=True

    res=mask(val,lam,theta,mode=(mode if mode is not None else "E"),**mask_cfg)
    if isinstance(res,tuple) and len(res)==5:
        lam2,th2,m,thr,report=res
    else:
        lam2,th2,m,thr=res
        report=None

    lines,_=draw(ax,lam2,th2,m,**draw_cfg)
    if return_report:
        return lam2,th2,m,thr,lines,report
    return lam2,th2,m,thr,lines
