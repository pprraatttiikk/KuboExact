
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pathlib import Path


def out_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def setup_plots():
    plt.rcParams.update({
        "text.usetex": False,
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def prettify(ax):
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(direction="out", length=4, width=1.0)
    ax.margins(x=0)


def savefig(fig, name, dpi=500):
    p = out_dir() / f"{name}.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    print("Saved:", p)


def popcount(x: int) -> int:
    return x.bit_count()


def occ(s: int, m: int) -> int:
    return (s >> m) & 1


def setbit(s: int, m: int, v: int) -> int:
    if v:
        return s | (1 << m)
    return s & ~(1 << m)


def site_occ(s: int, i: int) -> int:
    return occ(s, 2*i) | occ(s, 2*i + 1)


def between_parity(s: int, a: int, b: int) -> int:
    if a == b:
        return 0
    lo, hi = (a, b) if a < b else (b, a)
    mask = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
    return popcount(s & mask)


def hop(s: int, src: int, dst: int):
    if occ(s, src) == 0 or occ(s, dst) == 1:
        return None
    nb = between_parity(s, src, dst)
    sign = -1.0 if (nb % 2 == 1) else +1.0
    s2 = s
    s2 = setbit(s2, src, 0)
    s2 = setbit(s2, dst, 1)
    return s2, sign


def bonds_ring(L: int):
    return [(i, (i + 1) % L) for i in range(L)]


def make_basis(L: int, Nup: int, Ndn: int):
    modes = 2 * L
    N = Nup + Ndn
    st = []
    for s in range(1 << modes):
        if popcount(s) != N:
            continue
        ok = True
        nu = 0
        nd = 0
        for i in range(L):
            u = occ(s, 2*i)
            d = occ(s, 2*i + 1)
            if u & d:
                ok = False
                break
            nu += u
            nd += d
        if ok and (nu == Nup) and (nd == Ndn):
            st.append(s)
    st = np.array(st, dtype=np.int64)
    idx = {int(s): i for i, s in enumerate(st)}
    return st, idx


def H_tJ(L: int, states, index, t: float, J: float):
    dim = len(states)
    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)

    bd = bonds_ring(L)

    for a, s in enumerate(states):
        s = int(s)

        for (i, j) in bd:
            for spn in (0, 1):
                mi = 2*i + spn
                mj = 2*j + spn

                if occ(s, mj) == 1 and site_occ(s, i) == 0:
                    out = hop(s, mj, mi)
                    if out is not None:
                        s2, sg = out
                        b = index.get(s2, None)
                        if b is not None:
                            add(a, b, -t * sg)

                if occ(s, mi) == 1 and site_occ(s, j) == 0:
                    out = hop(s, mi, mj)
                    if out is not None:
                        s2, sg = out
                        b = index.get(s2, None)
                        if b is not None:
                            add(a, b, -t * sg)

        for (i, j) in bd:
            ui = occ(s, 2*i)
            di = occ(s, 2*i + 1)
            uj = occ(s, 2*j)
            dj = occ(s, 2*j + 1)

            ni = ui | di
            nj = uj | dj
            if (ni == 0) or (nj == 0):
                continue

            if ui == 1 and dj == 1:
                add(a, a, -0.5 * J)
                s2 = s
                s2 = setbit(s2, 2*i, 0);     s2 = setbit(s2, 2*i + 1, 1)
                s2 = setbit(s2, 2*j, 1);     s2 = setbit(s2, 2*j + 1, 0)
                b = index.get(s2, None)
                if b is not None:
                    add(a, b, +0.5 * J)

            elif di == 1 and uj == 1:
                add(a, a, -0.5 * J)
                s2 = s
                s2 = setbit(s2, 2*i, 1);     s2 = setbit(s2, 2*i + 1, 0)
                s2 = setbit(s2, 2*j, 0);     s2 = setbit(s2, 2*j + 1, 1)
                b = index.get(s2, None)
                if b is not None:
                    add(a, b, +0.5 * J)

    H = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim))
    H = 0.5 * (H + H.T.conjugate())
    return H.tocsr()


def J_charge(L: int, states, index, t: float):
    dim = len(states)
    rows, cols, vals = [], [], []

    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)

    bd = bonds_ring(L)

    for a, s in enumerate(states):
        s = int(s)
        for (i, j) in bd:
            for spn in (0, 1):
                mi = 2*i + spn
                mj = 2*j + spn

                if occ(s, mj) == 1 and site_occ(s, i) == 0:
                    out = hop(s, mj, mi)
                    if out is not None:
                        s2, sg = out
                        b = index.get(s2, None)
                        if b is not None:
                            add(a, b, 1j * t * sg)

                if occ(s, mi) == 1 and site_occ(s, j) == 0:
                    out = hop(s, mi, mj)
                    if out is not None:
                        s2, sg = out
                        b = index.get(s2, None)
                        if b is not None:
                            add(a, b, -1j * t * sg)

    Jop = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim))
    Jop = 0.5 * (Jop + Jop.getH())
    return Jop.tocsr()


def lorentz(x, eta):
    return (1.0 / np.pi) * (eta / (x*x + eta*eta))


def sigma_lehmann(E, V, Jop, L, beta, wmax, nw, eta, tol=1e-12):
    E = np.asarray(E)
    dim = E.size

    w = np.linspace(0.0, wmax, nw)
    dw = w[1] - w[0]

    E0 = float(E[0])
    bol = np.exp(-beta * (E - E0))
    Z = float(np.sum(bol))

    JV = Jop.dot(V)
    Je = V.conj().T @ JV
    JJ = np.abs(Je)**2

    sticks = np.zeros_like(w, dtype=float)

    for n in range(dim):
        bn = bol[n]
        dE = E - E[n]
        msk = dE > 1e-14
        if not np.any(msk):
            continue
        JJmn = JJ[:, n]
        msk = msk & (JJmn > tol)
        if not np.any(msk):
            continue
        mids = np.where(msk)[0]
        dE2 = dE[mids]
        bm = bol[mids]
        wt = (np.pi / (L * Z)) * (bn - bm) / dE2 * JJmn[mids]

        bins = np.rint(dE2 / dw).astype(int)
        bins = np.clip(bins, 0, nw - 1)
        np.add.at(sticks, bins, wt / dw)

    shift = (np.arange(nw) - nw//2) * dw
    K = lorentz(shift, eta)
    K = np.fft.ifftshift(K)
    sig = np.real(np.fft.ifft(np.fft.fft(sticks) * np.fft.fft(K))) * dw
    sig[0] = sig[1]
    return w, sig


def solve(L, Nup, Ndn, t, J):
    st, idx = make_basis(L, Nup, Ndn)
    H = H_tJ(L, st, idx, t=t, J=J)
    Jop = J_charge(L, st, idx, t=t)
    E, V = np.linalg.eigh(H.toarray())
    return st, Jop, E, V


def sigmas_for_Ts(L, Nup, Ndn, t, J, Ts, wmax, nw, eta):
    st, Jop, E, V = solve(L, Nup, Ndn, t, J)
    out = []
    for T in Ts:
        w, s = sigma_lehmann(E, V, Jop, L, beta=1.0/T, wmax=wmax, nw=nw, eta=eta)
        out.append(s)
    return st, w, np.array(out)


def sector_for_holes(L, h):
    Ne = L - h
    if Ne < 0:
        raise ValueError("bad hole number")
    if Ne == 0:
        return None
    if Ne % 2 == 0:
        return Ne, Ne//2, Ne//2
    return Ne, (Ne+1)//2, (Ne-1)//2


def main():
    setup_plots()

    L = 8
    t = 1.0
    J0 = 0.4

    eta = 0.08
    nw = 1400

    Ts_overlay = [0.15, 0.30, 0.60, 1.00]
    Ts_heat = np.linspace(0.12, 1.2, 18).tolist()


    sec1 = (4, 3)

    st_tmp, Jop_tmp, E_tmp, V_tmp = solve(L, sec1[0], sec1[1], t=t, J=J0)
    bw_ref = float(E_tmp[-1] - E_tmp[0])


    wmax = bw_ref + 2.0


    st1, wA, sigA = sigmas_for_Ts(L, sec1[0], sec1[1], t=t, J=J0,
                                  Ts=Ts_overlay, wmax=wmax, nw=nw, eta=eta)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for k, T in enumerate(Ts_overlay):
        ax.plot(wA, sigA[k], label=rf"$T/t={T:g}$")
    ax.set_xlabel(r"$\omega/t$")
    ax.set_ylabel(r"$\sigma(\omega)$")
    ax.set_xlim(0, wmax)
    prettify(ax)
    ax.legend(frameon=False)
    savefig(fig, "A_sigma_vs_omega_T_overlay_1hole")


    st1h, wC, sigC = sigmas_for_Ts(L, sec1[0], sec1[1], t=t, J=J0,
                                  Ts=Ts_heat, wmax=wmax, nw=nw, eta=eta)
    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    im = ax.imshow(sigC, aspect="auto", origin="lower",
                   extent=[wC[0], wC[-1], Ts_heat[0], Ts_heat[-1]])
    ax.set_xlabel(r"$\omega/t$")
    ax.set_ylabel(r"$T/t$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$\sigma(\omega)$")
    savefig(fig, "C_heatmap_sigma_omega_T_1hole")

 
    T_fix = 0.30
    beta = 1.0 / T_fix

    all_curves = []

    for h in range(0, L): 
        sec = sector_for_holes(L, h)
        if sec is None:
            continue
        Ne, Nup, Ndn = sec

        st, Jop, E, V = solve(L, Nup, Ndn, t=t, J=J0)
        w, s = sigma_lehmann(E, V, Jop, L, beta=beta, wmax=wmax, nw=nw, eta=eta)

        all_curves.append((h, w, s))

        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        ax.plot(w, s)
        ax.set_xlabel(r"$\omega/t$")
        ax.set_ylabel(r"$\sigma(\omega)$")
        ax.set_xlim(0, wmax)
        prettify(ax)

        ax.text(
            0.98, 0.98,
            rf"$h={h},\; N_e={Ne},\; (N_\uparrow,N_\downarrow)=({Nup},{Ndn}),\; \mathrm{{dim}}={len(st)}$",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=11
        )
        savefig(fig, f"holes_{h:02d}_sigma_vs_omega_T0p30")


    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for (h, w, s) in all_curves:
        ax.plot(w, s, label=rf"$h={h}$")
    ax.set_xlabel(r"$\omega/t$")
    ax.set_ylabel(r"$\sigma(\omega)$")
    ax.set_xlim(0, wmax)
    prettify(ax)
    ax.legend(ncols=4, frameon=False, handlelength=2.0, columnspacing=1.0)
    savefig(fig, "holes_overlay_sigma_vs_omega_T0p30")


if __name__ == "__main__":
    main()