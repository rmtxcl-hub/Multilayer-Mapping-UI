import numpy as np
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes, binary_closing
import simplepeak as sp

def _as_axes(A, lam, theta):
    lam=np.asarray(lam,float).ravel(); th=np.asarray(theta,float).ravel(); Z=np.asarray(A)
    if Z.shape==(th.size,lam.size): return Z,lam,th
    if Z.shape==(lam.size,th.size): return Z.T,lam,th
    raise ValueError("shape mismatch")

def _nearest_idx(c, x):
    c=np.asarray(c,float).ravel()
    return int(np.argmin(np.abs(c-float(x))))

def _smooth1d(w, k):
    w=np.asarray(w,float); k=int(k)
    return w if k<=1 else np.convolve(w, np.ones(k,float)/k, mode="same")

def _otsu_1d_threshold(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if x.size==0: return 0.0
    xi=np.round(x).astype(int); xi=xi[xi>0]
    if xi.size==0: return 0.0
    mx=int(xi.max())
    h=np.bincount(xi, minlength=mx+1).astype(float)
    p=h/(h.sum()+1e-300)
    w=np.cumsum(p); mu=np.cumsum(p*np.arange(p.size)); mu_t=mu[-1]
    den=w*(1.0-w); den[den==0]=np.nan
    sb=(mu_t*w-mu)**2/den
    return float(np.nanargmax(sb))

def _span_otsu(w, idx, *, smooth=5, pad=1, min_keep=3):
    ws=_smooth1d(w,smooth); n=ws.size
    idx=int(np.clip(idx,0,n-1))
    if ws[idx]<=0: return idx,idx
    T=_otsu_1d_threshold(ws)
    good=(ws>=T)
    if not good[idx]: good[idx]=True
    L=idx
    while L-1>=0 and good[L-1]: L-=1
    R=idx
    while R+1<n and good[R+1]: R+=1
    L=max(0,L-int(pad)); R=min(n-1,R+int(pad))
    if (R-L+1)<min_keep:
        L=max(0,idx-min_keep//2); R=min(n-1,L+min_keep-1)
        L=max(0,R-min_keep+1)
    return int(L), int(R)

def peak_masks(val, lam, theta, mask, *, mode="E", peak=None, thr_used=None,
               close=2, fill=True, return_debug=False, **_):
    Z,lam,th=_as_axes(val,lam,theta)
    M=np.asarray(mask,bool)
    if M.shape==(lam.size,th.size): M=M.T
    if M.shape!=Z.shape: raise ValueError("mask shape mismatch")
    if peak is None: peak="high" if str(mode).upper() in ("E","A") else "low"

    st=generate_binary_structure(2,2)
    lab,n=label(M.astype(np.uint8), structure=st)
    vmeta,hmeta,_=sp.locate_1d_peaks(Z, lam, th, M, mode=mode, peak=peak, thr_used=thr_used, return_meta=True)

    MV=np.zeros_like(M,bool); MH=np.zeros_like(M,bool)

    for o in vmeta:
        x=float(o["x"]); bid=int(o.get("blob",0))
        if bid<=0 or bid>n: continue
        comp=(lab==bid)
        j=_nearest_idx(lam,x)
        w=comp.sum(axis=0)               # projection along theta -> lambda profile
        L,R=_span_otsu(w,j)
        reg=comp.copy(); reg[:,:L]=False; reg[:,R+1:]=False
        if close and close>0: reg=binary_closing(reg, structure=st, iterations=int(close))
        if fill: reg=binary_fill_holes(reg)
        MV |= (reg & comp)

    for o in hmeta:
        y=float(o["y"]); bid=int(o.get("blob",0))
        if bid<=0 or bid>n: continue
        comp=(lab==bid)
        i=_nearest_idx(th,y)
        w=comp.sum(axis=1)               # projection along lambda -> theta profile
        L,R=_span_otsu(w,i)
        reg=comp.copy(); reg[:L,:]=False; reg[R+1:,:]=False
        if close and close>0: reg=binary_closing(reg, structure=st, iterations=int(close))
        if fill: reg=binary_fill_holes(reg)
        MH |= (reg & comp)

    dbg=dict(v_peaks=vmeta, h_peaks=hmeta)
    return (MV, MH, dbg) if return_debug else (MV, MH)
