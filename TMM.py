import numpy as np

def _is_cupy(xp):
    return xp is not np and getattr(xp, "__name__", "").startswith("cupy")

class TMM:
    def __init__(self, tol_im=0.0, tol_re=0.0, clip_im_delta=60.0, xp=None):
        self.tol_im = float(tol_im)
        self.tol_re = float(tol_re)
        self.clip_im_delta = float(clip_im_delta)
        self.xp = np if xp is None else xp
        self._cupy = _is_cupy(self.xp)

    def _any(self, x):
        if not self._cupy:
            return bool(np.any(x))
        return bool(self.xp.asnumpy(self.xp.any(x)))

    def ct(self, n0, s0, nj):
        xp = self.xp
        sj = (n0 * s0) / nj
        cj = xp.sqrt(1.0 - sj * sj)
        cj = xp.where(xp.imag(nj * cj) < 0, -cj, cj)
        if self._any(xp.imag(nj * cj) < -self.tol_im):
            raise ValueError("Im(n*cosθ)<0 (branch/decay sanity fail)")
        return cj

    def q(self, nj, cj, pol):
        xp = self.xp
        qj = nj * cj if pol == "s" else nj / cj
        if self._any(xp.real(qj) < -self.tol_re):
            raise ValueError("Re(q)<0 (passivity/admittance sanity fail)")
        return qj

    def cs(self, delta):
        xp = self.xp
        a = xp.real(delta)
        b = xp.imag(delta)
        b = xp.clip(b, -self.clip_im_delta, self.clip_im_delta)
        ep = xp.exp(1j * a - b)
        em = xp.exp(-1j * a + b)
        return 0.5 * (ep + em), (ep - em) / (2j)

    def _A_from_k0(self, k0, n0, s0, nj, dj, pol):
        cj = self.ct(n0, s0, nj)
        qj = self.q(nj, cj, pol)
        delta = k0 * nj * float(dj) * cj
        cd, sd = self.cs(delta)
        return cd, -1j * sd / qj, -1j * qj * sd, cd

    def A(self, lam, n0, s0, nj, dj, pol):
        xp = self.xp
        lam = xp.asarray(lam, float)
        k0 = (2.0 * xp.pi) / lam
        return self._A_from_k0(k0, n0, s0, nj, dj, pol)

    def M(self, lam, theta, n0, nL, dL, pol):
        xp = self.xp
        lam = xp.asarray(lam, float)
        th = xp.asarray(theta, float)
        k0 = (2.0 * xp.pi) / lam

        if th.ndim == 0:
            s0 = xp.sin(float(th))
            M11 = xp.ones_like(lam, complex)
            M12 = xp.zeros_like(lam, complex)
            M21 = xp.zeros_like(lam, complex)
            M22 = xp.ones_like(lam, complex)
            for nj, dj in zip(nL, dL):
                A11, A12, A21, A22 = self._A_from_k0(k0, n0, s0, nj, float(dj), pol)
                T11 = M11 * A11 + M12 * A21
                T12 = M11 * A12 + M12 * A22
                T21 = M21 * A11 + M22 * A21
                T22 = M21 * A12 + M22 * A22
                M11, M12, M21, M22 = T11, T12, T21, T22
            return M11, M12, M21, M22

        lam2 = lam[:, None]
        k02 = k0[:, None]
        th2 = th[None, :]
        s0 = xp.sin(th2)
        M11 = xp.ones((lam2.shape[0], th2.shape[1]), dtype=complex)
        M12 = xp.zeros_like(M11)
        M21 = xp.zeros_like(M11)
        M22 = xp.ones_like(M11)
        for nj, dj in zip(nL, dL):
            nj2 = nj[:, None]
            A11, A12, A21, A22 = self._A_from_k0(k02, n0, s0, nj2, float(dj), pol)
            T11 = M11 * A11 + M12 * A21
            T12 = M11 * A12 + M12 * A22
            T21 = M21 * A11 + M22 * A21
            T22 = M21 * A12 + M22 * A22
            M11, M12, M21, M22 = T11, T12, T21, T22
        return M11, M12, M21, M22

    def yin(self, lam, theta, n0, ns, nL, dL, pol):
        xp = self.xp
        lam = xp.asarray(lam, float)
        th = xp.asarray(theta, float)

        if th.ndim == 0:
            s0 = xp.sin(float(th))
            c0 = self.ct(n0, s0, n0)
            q0 = self.q(n0, c0, pol)
            cs = self.ct(n0, s0, ns)
            qs = self.q(ns, cs, pol)
            M11, M12, M21, M22 = self.M(lam, th, n0, nL, dL, pol)
            Yin = (M21 + M22 * qs) / (M11 + M12 * qs)
            return Yin, q0, qs, (M11, M12, M21, M22)

        th2 = th[None, :]
        s0 = xp.sin(th2)
        c0 = self.ct(n0, s0, n0)
        q0 = self.q(n0, c0, pol)
        ns2 = ns[:, None]
        cs = self.ct(n0, s0, ns2)
        qs = self.q(ns2, cs, pol)
        M11, M12, M21, M22 = self.M(lam, th, n0, nL, dL, pol)
        Yin = (M21 + M22 * qs) / (M11 + M12 * qs)
        return Yin, q0, qs, (M11, M12, M21, M22)

    def r(self, lam, theta, n0, ns, nL, dL, pol):
        xp = self.xp
        lam = xp.asarray(lam, float)
        th = xp.asarray(theta, float)

        if th.ndim == 0:
            s0 = xp.sin(float(th))
            c0 = self.ct(n0, s0, n0)
            q0 = self.q(n0, c0, pol)
            cs = self.ct(n0, s0, ns)
            qs = self.q(ns, cs, pol)
            M11, M12, M21, M22 = self.M(lam, th, n0, nL, dL, pol)
            Yin = (M21 + M22 * qs) / (M11 + M12 * qs)
            return (q0 - Yin) / (q0 + Yin)

        th2 = th[None, :]
        s0 = xp.sin(th2)
        c0 = self.ct(n0, s0, n0)
        q0 = self.q(n0, c0, pol)
        ns2 = ns[:, None]
        cs = self.ct(n0, s0, ns2)
        qs = self.q(ns2, cs, pol)
        M11, M12, M21, M22 = self.M(lam, th, n0, nL, dL, pol)
        Yin = (M21 + M22 * qs) / (M11 + M12 * qs)
        return (q0 - Yin) / (q0 + Yin)

    def rt(self, lam, theta, n0, ns, nL, dL, pol):
        xp = self.xp
        lam = xp.asarray(lam, float)
        th = xp.asarray(theta, float)

        if th.ndim == 0:
            s0 = xp.sin(float(th))
            c0 = self.ct(n0, s0, n0)
            q0 = self.q(n0, c0, pol)
            cs = self.ct(n0, s0, ns)
            qs = self.q(ns, cs, pol)
            M11, M12, M21, M22 = self.M(lam, th, n0, nL, dL, pol)
            den = q0 * (M11 + M12 * qs) + (M21 + M22 * qs)
            r = (q0 * (M11 + M12 * qs) - (M21 + M22 * qs)) / den
            t = (2.0 * q0) / den
            return r, t

        th2 = th[None, :]
        s0 = xp.sin(th2)
        c0 = self.ct(n0, s0, n0)
        q0 = self.q(n0, c0, pol)
        ns2 = ns[:, None]
        cs = self.ct(n0, s0, ns2)
        qs = self.q(ns2, cs, pol)
        M11, M12, M21, M22 = self.M(lam, th, n0, nL, dL, pol)
        den = q0 * (M11 + M12 * qs) + (M21 + M22 * qs)
        r = (q0 * (M11 + M12 * qs) - (M21 + M22 * qs)) / den
        t = (2.0 * q0) / den
        return r, t