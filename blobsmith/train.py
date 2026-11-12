"""The loop: paint the blobs, compare with the photo, correct every blob a little, again.

Each blob is nine numbers: x, y, width, height, angle, red, green, blue, strength.
"""
import json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .render import render

LEARNING_RATES = {"pos": 0.35, "ls": 0.03, "th": 0.03, "col": 0.012, "op": 0.05}
GROW_EVERY = 40  # steps between rounds of adding blobs


class Adam:
    """Adam, written out by hand so blobs can be added and removed mid-run."""

    def __init__(self, P, lrs):
        self.P, self.lrs, self.t = P, lrs, 0
        self.m = {k: torch.zeros_like(v) for k, v in P.items()}
        self.v = {k: torch.zeros_like(v) for k, v in P.items()}

    def step(self, grads, mult):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mh = self.m[k] / (1 - 0.9 ** min(self.t, 200))
            vh = self.v[k] / (1 - 0.999 ** min(self.t, 5000))
            self.P[k].data -= self.lrs[k] * mult * mh / (vh.sqrt() + 1e-8)

    def keep(self, mask):
        for k in self.P:
            self.P[k] = self.P[k].data[mask].requires_grad_(True)
            self.m[k], self.v[k] = self.m[k][mask], self.v[k][mask]

    def add(self, new):
        for k in self.P:
            self.P[k] = torch.cat([self.P[k].data, new[k]]).requires_grad_(True)
            z = torch.zeros_like(new[k])
            self.m[k], self.v[k] = torch.cat([self.m[k], z]), torch.cat([self.v[k], z])


def to_image(t):
    return Image.fromarray((t.detach().clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8))


def psnr(a, b):
    return -10 * math.log10(max(F.mse_loss(a.clamp(0, 1), b).item(), 1e-12))


def load_photo(path, size):
    """Open a photo and shrink it so its longest side is at most `size` pixels."""
    im = Image.open(path).convert("RGB")
    if max(im.size) > size:
        k = size / max(im.size)
        im = im.resize((max(8, round(im.width * k)), max(8, round(im.height * k))), Image.LANCZOS)
    return im


def fit(photo, out, size=400, steps=2400, max_blobs=36000, start_blobs=1500, crop=200,
        seed=0, device="cpu", say=print):
    """Fit blobs to `photo` and record the run in the folder `out`. Returns (blobs, meta)."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    for old in os.listdir(os.path.join(out, "frames")):
        os.remove(os.path.join(out, "frames", old))

    im = load_photo(photo, size)
    im.save(os.path.join(out, "photo.png"))
    target = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).to(dev)
    H, W, _ = target.shape
    side = math.sqrt(H * W)
    pyramid = {1.0: target}
    for s in (0.25, 0.5):
        hw = (max(4, round(H * s)), max(4, round(W * s)))
        pyramid[s] = F.interpolate(target.permute(2, 0, 1)[None], hw, mode="area")[0].permute(1, 2, 0)

    # random confetti to start from
    n = start_blobs
    P = {
        "pos": torch.rand(n, 2, device=dev) * torch.tensor([W, H], device=dev),
        "ls": torch.log(torch.rand(n, 2, device=dev) * 0.035 * side + 0.012 * side),
        "th": torch.rand(n, device=dev) * math.pi,
        "col": torch.rand(n, 3, device=dev),
        "op": torch.full((n,), -1.0, device=dev),
    }
    for k in P:
        P[k].requires_grad_(True)
    opt = Adam(P, LEARNING_RATES)
    P = opt.P

    T = steps
    draft_until, half_until, grow_until = math.ceil(T * 0.06), math.ceil(T * 0.22), int(T * 0.72)
    shots = sorted(set([s for s in (0, 2, 5, 9, 15, 25, 40, 60, 80) if s < T]
                       + list(range(100, T, max(20, T // 110))) + [T - 1]))
    record, t0 = [], time.time()
    for step in range(T):
        s = 0.25 if step < draft_until else 0.5 if step < half_until else 1.0
        tgt = pyramid[s]; h, w = tgt.shape[:2]
        if s == 1.0 and crop and crop < min(H, W):
            # look at one random patch per step, painting only the blobs that reach it
            x0, y0 = np.random.randint(0, W - crop + 1), np.random.randint(0, H - crop + 1)
            with torch.no_grad():
                reach = 3.0 * torch.exp(P["ls"]).max(dim=1).values + 1
                near = (P["pos"][:, 0] > x0 - reach) & (P["pos"][:, 0] < x0 + crop + reach) \
                     & (P["pos"][:, 1] > y0 - reach) & (P["pos"][:, 1] < y0 + crop + reach)
            sub = {k: v[near] for k, v in P.items()}
            sub["pos"] = sub["pos"] - torch.tensor([x0, y0], device=dev)
            diff = render(sub, crop, crop, 1.0) - tgt[y0:y0 + crop, x0:x0 + crop]
        else:
            diff = render(P, h, w, s) - tgt
        loss = diff.abs().mean() * 0.8 + (diff * diff).mean() * 4.0
        grads = dict(zip(P.keys(), torch.autograd.grad(loss, list(P.values()))))
        grads["pos"] = grads["pos"] * s  # keep position steps the same at every resolution
        opt.step(grads, 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / T)))
        with torch.no_grad():
            P["ls"].data.clamp_(math.log(0.25), math.log(0.6 * max(H, W)))
            thin = P["ls"].data.min(dim=1, keepdim=True).values  # no slivers: at most 5:1
            P["ls"].data.copy_(torch.minimum(P["ls"].data, thin + math.log(5.0)))
            P["col"].data.clamp_(-0.05, 1.05)

        if step in shots:
            with torch.no_grad():
                full = render(P, H, W, 1.0)
            to_image(full).save(os.path.join(out, "frames", f"{len(record):04d}.png"))
            record.append([step, P["pos"].shape[0], round(psnr(full, target), 3)])
            say(f"step {step + 1:5d}/{T}   blobs {P['pos'].shape[0]:6d}   "
                f"match {record[-1][2]:5.2f} dB   {time.time() - t0:4.0f}s")

        # every so often: drop dead blobs, add new ones where the picture is most wrong
        if step % GROW_EVERY == GROW_EVERY - 1 and step < grow_until:
            with torch.no_grad():
                alive = (torch.sigmoid(P["op"]) > 0.01) & (P["pos"][:, 0] > -8) & (P["pos"][:, 0] < W + 8) \
                      & (P["pos"][:, 1] > -8) & (P["pos"][:, 1] < H + 8)
                if not alive.all():
                    opt.keep(alive); P = opt.P
                goal = start_blobs * (max_blobs / start_blobs) ** min(1.0, (step + 1) / grow_until)
                add = int(min(max_blobs, goal)) - P["pos"].shape[0]
                if add > 0:
                    err = (render(P, h, w, s) - tgt).abs().sum(-1)
                    err = F.avg_pool2d(err[None, None], 3, 1, 1)[0, 0].reshape(-1) ** 1.5
                    pick = torch.multinomial(err + 1e-9, add, replacement=True)
                    py, px = torch.div(pick, w, rounding_mode="floor").float(), (pick % w).float()
                    width = (0.9 / s) * (0.7 + 0.8 * torch.rand(add, 1, device=dev))
                    opt.add({
                        "pos": torch.stack([px, py], 1) / s + (torch.rand(add, 2, device=dev) - 0.5) / s,
                        "ls": torch.log(width.expand(add, 2) * (0.85 + 0.3 * torch.rand(add, 2, device=dev))),
                        "th": torch.rand(add, device=dev) * math.pi,
                        "col": tgt.reshape(-1, 3)[pick],
                        "op": torch.zeros(add, device=dev),
                    })
                    P = opt.P

    with torch.no_grad():
        final = render(P, H, W, 1.0)
    meta = {
        "photo": os.path.basename(photo), "width": W, "height": H,
        "steps": T, "start_blobs": start_blobs, "blobs": int(P["pos"].shape[0]),
        "draft_until": draft_until, "half_until": half_until, "grow_until": grow_until,
        "grow_every": GROW_EVERY, "crop": crop if crop and crop < min(H, W) else 0,
        "match_db": round(psnr(final, target), 2),
        "mean_error": round((final.clamp(0, 1) - target).abs().mean().item() * 255, 2),
        "seconds": round(time.time() - t0), "seed": seed, "frames": record,
    }
    np.savez_compressed(os.path.join(out, "blobs.npz"), **{k: v.detach().cpu().numpy() for k, v in P.items()})
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    return P, meta
