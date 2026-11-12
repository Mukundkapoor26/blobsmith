"""Command line:  python -m blobsmith PHOTO  (see --help for the knobs)."""
import argparse, os

from . import media
from .train import fit


def main():
    ap = argparse.ArgumentParser(
        prog="python -m blobsmith",
        description="Rebuild a photo out of soft blobs of colour, and record how it happened.")
    ap.add_argument("photo", help="any image file Pillow can open")
    ap.add_argument("--out", help="folder for the results (default: runs/<photo name>)")
    ap.add_argument("--size", type=int, default=400, help="longest side to work at, in pixels (default 400)")
    ap.add_argument("--steps", type=int, default=2400, help="training rounds (default 2400)")
    ap.add_argument("--blobs", type=int, default=36000, help="most blobs it may use (default 36000)")
    ap.add_argument("--start-blobs", type=int, default=1500, help="random blobs to begin with (default 1500)")
    ap.add_argument("--crop", type=int, default=200, help="patch size for full-size rounds, 0 = whole picture")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu (default), mps or cuda")
    ap.add_argument("--quick", action="store_true", help="a 200-round, 4000-blob trial run (about a minute)")
    ap.add_argument("--no-media", action="store_true", help="only train; skip the GIF, video and viewer")
    a = ap.parse_args()
    if a.quick:
        a.steps, a.blobs = 200, 4000
    out = a.out or os.path.join("runs", os.path.splitext(os.path.basename(a.photo))[0])

    _, meta = fit(a.photo, out, size=a.size, steps=a.steps, max_blobs=a.blobs, start_blobs=a.start_blobs,
                  crop=a.crop, seed=a.seed, device=a.device)
    print(f"\n{meta['blobs']:,} blobs, off by {meta['mean_error']} out of 255 per colour value on average "
          f"({meta['match_db']} dB), in {meta['seconds']}s")
    if not a.no_media:
        print("making pictures, GIF, video and viewer...")
        for name in media.build(out):
            print("  " + os.path.join(out, name))
        print("\nOpen " + os.path.join(out, "viewer", "index.html") + " in a browser to replay the run.")


if __name__ == "__main__":
    main()
