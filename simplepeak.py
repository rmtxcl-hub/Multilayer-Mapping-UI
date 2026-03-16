import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label, generate_binary_structure
from scipy.signal import find_peaks

def _as_axes(val, lam, theta):
    lam=np.asarray(lam,float).ravel(); th=np.asarray(theta,float).ravel(); Z=np.asarray(val,float)
    if Z.shape==(th.size,lam.size): return Z,lam,th
    if Z.shape==(lam.size,th.size): return Z.T,lam,th
    raise ValueError("shape mismatch")

def _interp_idx_to_coord(coord, idx):
    x=np.arange(coord.size, dtype=float)
    return float(np.interp(float(idx), x, coord))

def _valley_bounds(sig_norm, peaks, k, beta=0.25):
    n=sig_norm.size; i0=int(peaks[k]); pk=float(sig_norm[i0]); pk=pk if pk>0 else 1e-12
    thr=beta*pk
    if k==0:
        L=i0
        while L>0 and sig_norm[L]>=thr: L-=1
    else:
        a=int(peaks[k-1]); b=i0
        L=a if b-a<=1 else int(np.argmin(sig_norm[a:b+1])+a)
    if k==len(peaks)-1:
        R=i0
        while R<n-1 and sig_norm[R]>=thr: R+=1
    else:
        a=i0; b=int(peaks[k+1])
        R=b if b-a<=1 else int(np.argmin(sig_norm[a:b+1])+a)
    if L>R: L,R=R,L
    return int(max(0,L)), int(min(n-1,R))

def _area_median_idx(sig_raw, L, R):
    x=sig_raw[L:R+1].astype(float)
    x=np.where(np.isfinite(x), x, 0.0); x=np.maximum(x, 0.0)
    s=float(x.sum())
    if s<=0: return 0.5*(L+R)
    c=np.cumsum(x); j=int(np.searchsorted(c, 0.5*s, side="left"))
    return float(L+j)

def _merge_1d(vals, wts, gap):
    if not vals: return []
    a=sorted(zip(vals,wts),key=lambda t:t[0])
    out=[]; x,w=a[0]
    for xi,wi in a[1:]:
        if abs(xi-x)<=gap:
            s=w+wi+1e-300
            x=(x*w+xi*wi)/s; w=s
        else:
            out.append((float(x),float(w))); x,w=xi,wi
    out.append((float(x),float(w)))
    return out

def _dominant_blob(lab_block, w_block):
    L=lab_block.ravel().astype(np.int32)
    if L.size==0: return 0
    W=np.asarray(w_block,float).ravel()
    if W.size!=L.size: W=np.ones_like(L,float)
    m=L>0
    if not m.any(): return 0
    L=L[m]; W=W[m]
    s=np.bincount(L, weights=W)
    if s.size<=1: return 0
    return int(np.argmax(s[1:])+1)

def locate_1d_peaks(val, lam, theta, mask, *, mode="E", peak=None, thr_used=None,
                    sigW=0.8, sig1d=2.0,
                    prom_lam=0.10, prom_th=0.12, dist_lam=14, dist_th=10,
                    merge_lam=0.25, merge_th=1.5, topk_lam=None, topk_th=1,
                    span_pow_lam=2.0, span_pow_th=2.0, min_span_lam=0.25, min_span_th=0.40,
                    edge_lam=6, edge_th=3,
                    mid_beta=0.25, mid_mode="area_median",
                    return_meta=False):
    Z,lam,th=_as_axes(val,lam,theta)
    m=np.asarray(mask,bool)
    if m.shape==(lam.size,th.size): m=m.T
    if m.shape!=Z.shape: raise ValueError("mask shape mismatch")

    if peak is None: peak="high" if str(mode).upper() in ("E","A") else "low"
    V=-Z if peak=="low" else Z
    if sigW and sigW>0: V=gaussian_filter(V,float(sigW),mode="nearest")

    base=float(thr_used) if thr_used is not None else (float(np.nanmedian(V[m])) if np.any(m) else 0.0)
    W=np.where(m,np.maximum(V-base,0.0),0.0)

    b_lam=m.sum(0).astype(float); b_th=m.sum(1).astype(float)
    span_lam=b_lam/max(1,m.shape[0]); span_th=b_th/max(1,m.shape[1])
    span_lam=np.where(span_lam>=float(min_span_lam),span_lam**float(span_pow_lam),0.0)
    span_th=np.where(span_th>=float(min_span_th),span_th**float(span_pow_th),0.0)

    Bl=b_lam*span_lam; Bt=b_th*span_th
    if sig1d and sig1d>0:
        Bl=gaussian_filter1d(Bl,float(sig1d))
        Bt=gaussian_filter1d(Bt,float(sig1d))

    Bln=Bl/(Bl.max()+1e-300); Btn=Bt/(Bt.max()+1e-300)
    il,_=find_peaks(Bln,prominence=float(prom_lam),distance=int(dist_lam))
    it,_=find_peaks(Btn,prominence=float(prom_th),distance=int(dist_th))
    if edge_lam: il=il[(il>=int(edge_lam))&(il<lam.size-int(edge_lam))]
    if edge_th: it=it[(it>=int(edge_th))&(it<th.size-int(edge_th))]

    st=generate_binary_structure(2,2)
    lab,_=label(m.astype(np.uint8), structure=st)

    v_raw=[]
    for k in range(il.size):
        L,R=_valley_bounds(Bln,il,k,beta=float(mid_beta))
        if edge_lam:
            L=max(L,int(edge_lam)); R=min(R, lam.size-1-int(edge_lam))
            if L>=R: continue
        idx=0.5*(L+R) if str(mid_mode).lower()=="mid" else _area_median_idx(Bl,L,R)
        x=_interp_idx_to_coord(lam,idx)
        sc=float(W[:,L:R+1].sum())
        bid=_dominant_blob(lab[:,L:R+1], W[:,L:R+1])
        v_raw.append((x,sc,bid))

    h_raw=[]
    for k in range(it.size):
        L,R=_valley_bounds(Btn,it,k,beta=float(mid_beta))
        if edge_th:
            L=max(L,int(edge_th)); R=min(R, th.size-1-int(edge_th))
            if L>=R: continue
        idx=0.5*(L+R) if str(mid_mode).lower()=="mid" else _area_median_idx(Bt,L,R)
        y=_interp_idx_to_coord(th,idx)
        sc=float(W[L:R+1,:].sum())
        bid=_dominant_blob(lab[L:R+1,:], W[L:R+1,:])
        h_raw.append((y,sc,bid))

    def _merge_by_blob(raw, gap):
        by={}
        for x,w,b in raw: by.setdefault(int(b), []).append((float(x),float(w)))
        out=[]
        for b,arr in by.items():
            m=_merge_1d([x for x,_ in arr],[w for _,w in arr],float(gap))
            out += [(x,w,b) for x,w in m]
        return out

    v=_merge_by_blob(v_raw, merge_lam)
    h=_merge_by_blob(h_raw, merge_th)
    v=sorted(v,key=lambda t:-t[1]); h=sorted(h,key=lambda t:-t[1])
    if topk_lam is not None: v=v[:int(topk_lam)]
    if topk_th is not None: h=h[:int(topk_th)]

    if not return_meta:
        return [float(x) for x,_,_ in v],[float(y) for y,_,_ in h]
    return ([dict(x=float(x), w=float(w), blob=int(b)) for x,w,b in v],
            [dict(y=float(y), w=float(w), blob=int(b)) for y,w,b in h],
            dict(base=float(base)))
