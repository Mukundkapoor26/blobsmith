"""Turn a recorded run into things you can look at and share.

    python -m blobsmith.media runs/pearl-earring      # rebuild everything for an existing run

Writes into the run folder: blobs.png, blobs_3x.png, side_by_side.png, discs.png,
training.gif, film.mp4 (when ffmpeg is installed) and viewer/ (two web pages).
"""
import glob, json, math, os, shutil, subprocess, sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .render import render
from .train import to_image

BG, LINE, TEXT, DIM, ACCENT = (13, 13, 16), (38, 38, 46), (241, 239, 233), (143, 141, 150), (255, 176, 138)
HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = {  # (regular, bold) candidates per platform; first one that loads wins
    False: [("/System/Library/Fonts/HelveticaNeue.ttc", 0), ("/System/Library/Fonts/Helvetica.ttc", 0),
            ("DejaVuSans.ttf", 0), ("arial.ttf", 0), ("segoeui.ttf", 0)],
    True: [("/System/Library/Fonts/HelveticaNeue.ttc", 1), ("/System/Library/Fonts/Helvetica.ttc", 1),
           ("DejaVuSans-Bold.ttf", 0), ("arialbd.ttf", 0), ("segoeuib.ttf", 0)],
}


def font(size, bold=False):
    for path, index in FONTS[bold]:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def load_run(run):
    with open(os.path.join(run, "meta.json")) as f:
        meta = json.load(f)
    photo = Image.open(os.path.join(run, "photo.png")).convert("RGB")
    frames = [Image.open(p).convert("RGB") for p in sorted(glob.glob(os.path.join(run, "frames", "*.png")))]
    blobs = {k: torch.from_numpy(v) for k, v in np.load(os.path.join(run, "blobs.npz")).items()}
    return meta, photo, frames, blobs


def draw_discs(P, H, W, up, shrink=0.9):
    """The proof picture: every blob as a flat, hard-edged ellipse, largest first."""
    ss = 2
    im = Image.new("RGB", (W * up * ss, H * up * ss), (12, 12, 14))
    d = ImageDraw.Draw(im)
    pos = P["pos"].numpy() * up * ss
    sig = np.exp(P["ls"].numpy()) * up * ss * shrink
    th = P["th"].numpy()
    col = (P["col"].clamp(0, 1).numpy() * 255).astype(int)
    a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    for i in np.argsort(-(sig[:, 0] * sig[:, 1])):
        ex, ey = sig[i, 0] * ca, sig[i, 1] * sa
        c, s = math.cos(th[i]), math.sin(th[i])
        xs, ys = pos[i, 0] + c * ex - s * ey, pos[i, 1] + s * ex + c * ey
        d.polygon(list(zip(xs.tolist(), ys.tolist())), fill=tuple(col[i]))
    return im.resize((W * up, H * up), Image.LANCZOS)


def error_map(frame, photo, gain=5):
    """Where a frame differs from the photo, as a dark-to-bright heat picture."""
    d = np.abs(np.asarray(frame, dtype=np.float32) - np.asarray(photo, dtype=np.float32)).mean(-1)
    e = np.clip(d * gain, 0, 255) / 255
    rgb = np.stack([np.clip(e * 2.2, 0, 1), np.clip(e * 1.6 - 0.25, 0, 1),
                    np.clip(e * 1.2 - 0.15, 0, 1) * 0.8 + 0.06 * (1 - e)], -1)
    return Image.fromarray((rgb * 255).astype(np.uint8)), float(d.mean())


def fit_box(im, box):
    k = min(box / im.width, box / im.height)
    return im.resize((max(2, round(im.width * k)), max(2, round(im.height * k))), Image.LANCZOS)


def caption(meta, i, hold=False, discs=False):
    """One plain sentence for what is happening at frame i. Mirrored in viewer/film.html."""
    step = meta["frames"][i][0]
    if discs:
        return "Here are those same blobs with hard edges."
    if hold:
        return f"{meta['steps']:,} rounds in: {meta['blobs']:,} blobs, and no AI image model."
    if i < 5:
        return f"It begins as {meta['start_blobs']:,} blobs with random colours."
    if step < meta["half_until"]:
        return "Paint the blobs, compare with the photo, correct each one slightly."
    if step < meta["grow_until"]:
        return "Wherever the picture is still off, drop in a few smaller blobs."
    return "No more new blobs. The ones already there just fine-tune."


def make_stills(run, meta, photo, frames, blobs):
    H, W = meta["height"], meta["width"]
    with torch.no_grad():
        final = to_image(render(blobs, H, W, 1.0))
        big = to_image(render(blobs, H * 3, W * 3, 3.0))
    final.save(os.path.join(run, "blobs.png"))
    big.save(os.path.join(run, "blobs_3x.png"))
    discs = draw_discs(blobs, H, W, max(1, round(1600 / max(H, W))))
    discs.save(os.path.join(run, "discs.png"))

    # labelled pair, sized for a social post
    box, gap, pad, label = 700, 40, 70, 96
    a, b = fit_box(photo, box), fit_box(big, box)
    card = Image.new("RGB", (2 * a.width + gap + 2 * pad, a.height + pad + label + 34), BG)
    d = ImageDraw.Draw(card)
    f = font(30, bold=True)
    for k, (pic, text, colour) in enumerate([(a, "PHOTO", TEXT), (b, f"{meta['blobs']:,} BLOBS", ACCENT)]):
        x = pad + k * (a.width + gap)
        card.paste(pic, (x, pad), rounded_mask(pic.size, 18))
        d.text((x + (a.width - d.textlength(text, font=f)) / 2, pad + a.height + 34), text, font=f, fill=colour)
    card.save(os.path.join(run, "side_by_side.png"))
    return final, big, discs


def rounded_mask(size, radius):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius, fill=255)
    return m


def make_gif(run, photo, frames, discs, box=300):
    """Photo on the left, blobs forming on the right, ending on the disc view."""
    p = fit_box(photo, box)
    def pair(right):
        c = Image.new("RGB", (p.width * 2 + 8, p.height), BG)
        c.paste(p, (0, 0)); c.paste(right.resize(p.size, Image.LANCZOS), (p.width + 8, 0))
        return c.quantize(96, dither=Image.Dither.NONE)
    keep = [i for i in range(len(frames)) if i < 10 or i % 2 == 0 or i == len(frames) - 1]
    ims = [pair(frames[i]) for i in keep] + [pair(discs)]
    times = [220 if i < 10 else 110 for i in keep]
    times[-1] = 1800
    ims[0].save(os.path.join(run, "training.gif"), save_all=True, append_images=ims[1:],
                duration=times + [2200], loop=0, optimize=True)


def make_film(run, meta, photo, frames, big, discs):
    """A 1080p video of the run, laid out like viewer/film.html. Needs ffmpeg."""
    if not shutil.which("ffmpeg"):
        return False
    W, H, box, gap = 1920, 1080, 730, 48
    p = fit_box(photo, box)
    pw, ph = p.size
    x0, y0 = (W - 2 * pw - gap) // 2, 96 + (box - ph) // 2
    x1 = x0 + pw + gap
    mask = rounded_mask(p.size, 21)
    sized = lambda im: im.resize(p.size, Image.LANCZOS)
    stages, last, disc = [sized(f) for f in frames], sized(big), sized(discs)
    f_bold, f_reg, f_cap = font(28, True), font(26), font(36)
    base = Image.new("RGB", (W, H), BG)
    base.paste(p, (x0, y0), mask)
    ImageDraw.Draw(base).text((x0, y0 + ph + 22), "PHOTO", font=f_bold, fill=TEXT)

    n = len(frames)
    shots = [(i, 8 if i < 10 else 2, False, False) for i in range(n)]
    shots += [(n - 1, 54, True, False), (n - 1, 84, True, True), (n - 1, 42, True, False)]
    total, done = sum(s[1] for s in shots), 0
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium",
         "-movflags", "+faststart", os.path.join(run, "film.mp4")], stdin=subprocess.PIPE)
    bar_w = min(420, pw)
    for i, count, hold, as_discs in shots:
        step, blobs = meta["frames"][i][0], meta["frames"][i][1]
        for k in range(count):
            im = base.copy(); d = ImageDraw.Draw(im)
            pic = last if hold else stages[i]
            if as_discs:
                pic = Image.blend(last, disc, min(1, k / 15, (count - 1 - k) / 15))
            im.paste(pic, (x1, y0), mask)
            d.text((x1, y0 + ph + 22), f"{blobs:,} BLOBS", font=f_bold, fill=ACCENT)
            t = f"round {step + 1:,}"
            d.text((x1 + pw - d.textlength(t, font=f_reg), y0 + ph + 24), t, font=f_reg, fill=DIM)
            d.text((x0, y0 + ph + 92), caption(meta, i, hold, as_discs), font=f_cap, fill=TEXT)
            bx, by = x1 + pw - bar_w, y0 + ph + 110
            if d.textlength(caption(meta, i, hold, as_discs), font=f_cap) < bx - x0 - 30:
                d.rounded_rectangle((bx, by, bx + bar_w, by + 8), 4, fill=LINE)
                d.rounded_rectangle((bx, by, bx + max(8, int(bar_w * done / total)), by + 8), 4, fill=ACCENT)
            ff.stdin.write(im.tobytes()); done += 1
    ff.stdin.close()
    return ff.wait() == 0


def make_viewer(run, meta, photo, frames, discs):
    """Two self-contained web pages that replay the run. They open straight from disk."""
    out = os.path.join(run, "viewer")
    for sub in ("f", "e"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    photo.save(os.path.join(out, "photo.jpg"), quality=95)
    fit_box(discs, 1200).save(os.path.join(out, "discs.jpg"), quality=88)
    rows = []
    for i, (frame, (step, blobs, db)) in enumerate(zip(frames, meta["frames"])):
        frame.save(os.path.join(out, "f", f"{i}.jpg"), quality=88)
        heat, mean = error_map(frame, photo)
        heat.save(os.path.join(out, "e", f"{i}.jpg"), quality=85)
        rows.append([step, blobs, db, round(mean, 2)])
    data = {k: v for k, v in meta.items() if k != "frames"}
    data["frames"] = rows
    with open(os.path.join(out, "data.js"), "w") as f:
        f.write("window.RUN=" + json.dumps(data) + ";")
    for page in ("index.html", "film.html"):
        shutil.copy(os.path.join(HERE, "viewer", page), os.path.join(out, page))


def build(run, say=print):
    meta, photo, frames, blobs = load_run(run)
    final, big, discs = make_stills(run, meta, photo, frames, blobs)
    make_viewer(run, meta, photo, frames, discs)
    make_gif(run, photo, frames, discs)
    made = ["blobs.png", "blobs_3x.png", "side_by_side.png", "discs.png", "training.gif", "viewer/index.html"]
    if make_film(run, meta, photo, frames, big, discs):
        made.append("film.mp4")
    else:
        say("ffmpeg not found, so film.mp4 was skipped (everything else is there)")
    return made


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m blobsmith.media RUN_FOLDER")
    for name in build(sys.argv[1]):
        print(os.path.join(sys.argv[1], name))
