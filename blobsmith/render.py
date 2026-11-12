"""The painter: turns a list of blobs into a picture, in a way gradients can flow through.

A blob is an oriented 2D Gaussian. Every pixel's colour is the weighted average of the
blobs that reach it, so nothing has to be sorted and the whole thing is a few tensor ops.
"""
import torch
import torch.nn.functional as F

HALVES = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
MIN_SIGMA = 0.28  # px, at render resolution


def render(P, H, W, scale, shrink=1.0, chunk_px=3_000_000):
    """Paint blobs into an HxW picture. P is in full-res pixel units; scale maps to this render.

    Big soft blobs are painted on a coarser grid and upsampled (they carry no fine detail),
    so every blob costs about the same however large it is.
    """
    dev = P["pos"].device
    sig = (torch.exp(P["ls"]) * scale * shrink).clamp_min(MIN_SIGMA * min(1.0, shrink * 2))
    level = torch.floor(torch.log2(sig.min(dim=1).values.detach() / 1.25)).clamp(0, 5).long()
    f = (2.0 ** level)[:, None]
    mu = (P["pos"] * scale + 0.5) / f - 0.5
    sig = sig / f
    cos, sin = torch.cos(P["th"]), torch.sin(P["th"])
    # one row per blob: x y | 1/sx 1/sy | cos sin | alpha | r g b
    Q = torch.cat([mu, 1.0 / sig, cos[:, None], sin[:, None], torch.sigmoid(P["op"])[:, None], P["col"]], 1)
    # tight axis-aligned box around each (rotated) blob, snapped to a few sizes
    with torch.no_grad():
        ex = 3.0 * torch.sqrt((sig[:, 0] * cos) ** 2 + (sig[:, 1] * sin) ** 2)
        ey = 3.0 * torch.sqrt((sig[:, 0] * sin) ** 2 + (sig[:, 1] * cos) ** 2)
        edges = torch.tensor(HALVES, device=dev, dtype=ex.dtype)
        bx = torch.bucketize(torch.ceil(ex).clamp(2, HALVES[-1]), edges)
        by = torch.bucketize(torch.ceil(ey).clamp(2, HALVES[-1]), edges)
        key = level * 10000 + bx * 100 + by
        base = torch.round(mu).long()
        order = torch.argsort(key)
        keys, counts = torch.unique_consecutive(key[order], return_counts=True)
    bufs, start = {}, 0
    for k, n in zip(keys.tolist(), counts.tolist()):
        L, hx, hy = k // 10000, HALVES[k // 100 % 100], HALVES[k % 100]
        h, w = -(-H // 2 ** L), -(-W // 2 ** L)
        if L not in bufs:
            bufs[L] = torch.zeros(h * w, 4, device=dev)
        oy, ox = torch.meshgrid(torch.arange(-hy, hy + 1, device=dev),
                                torch.arange(-hx, hx + 1, device=dev), indexing="ij")
        ox, oy = ox.reshape(-1), oy.reshape(-1)
        per = max(1, chunk_px // ox.numel())
        for s in range(start, start + n, per):
            i = order[s:min(s + per, start + n)]
            q, bs = Q[i], base[i]
            mx, my, isx, isy, c, sn, al = [t[:, None] for t in q[:, :7].unbind(1)]
            px, py = bs[:, 0:1] + ox, bs[:, 1:2] + oy
            dx, dy = px - mx, py - my
            u = (c * dx + sn * dy) * isx
            v = (c * dy - sn * dx) * isy
            ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            wgt = al * torch.exp(-0.5 * (u * u + v * v)) * ok
            idx = (py.clamp(0, h - 1) * w + px.clamp(0, w - 1)).reshape(-1)
            val = torch.cat([wgt[..., None] * q[:, None, 7:10], wgt[..., None]], -1)
            bufs[L] = bufs[L].index_add(0, idx, val.reshape(-1, 4))
        start += n
    out = None
    for L, buf in bufs.items():
        buf = buf.reshape(-(-H // 2 ** L), -(-W // 2 ** L), 4)
        if L:
            buf = F.interpolate(buf.permute(2, 0, 1)[None], scale_factor=2 ** L, mode="bilinear",
                                align_corners=False)[0, :, :H, :W].permute(1, 2, 0)
        out = buf if out is None else out + buf
    return out[..., :3] / (out[..., 3:] + 1e-4)

