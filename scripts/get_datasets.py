"""Download and unpack the evaluation datasets into datasets/ (skips what is already there).

    python scripts/get_datasets.py bonn davis              # quick set, about 3.3 GB
    python scripts/get_datasets.py bonn davis sintel youtubematte    # full set, about 13 GB

bonn          Bonn RGB-D Dynamic: 5 test + 2 validation sequences   (~2.5 GB)
davis         DAVIS 2017 trainval 480p, only the val split is used  (~0.8 GB)
sintel        MPI Sintel depth + training images                    (~3.3 GB)
youtubematte  YouTubeMatte from Google Drive                        (~6.7 GB; Drive sometimes blocks it,
                                                                      then download it in a browser, see README)
Downloads resume if they stop halfway.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets"
BONN = "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_{}.zip"
BONN_SEQS = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous",       # test
             "moving_nonobstructing_box", "balloon_tracking"]                          # validation
FILES = {
    "davis": [("https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip",
               "davis/DAVIS-2017-trainval-480p.zip")],
    "sintel": [("https://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip", "sintel/depth.zip"),
               ("https://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip", "sintel/images.zip")],
    "bonn": [(BONN.format(s), f"bonn/rgbd_bonn_{s}.zip") for s in BONN_SEQS],
}
YTM_IDS = {"YouTubeMatte.zip": "1IEH0RaimT_hSp38AWF6wuwNJzzNSHpJ4",
           "first_frame_mask.zip": "1Zpa7SB7VZmkvRDiehVC-c_0dmFWXdfzK"}


def zip_ok(path):
    try:
        with zipfile.ZipFile(path) as z:
            return len(z.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False


def download(url, dst: Path):
    """Resumable download with a simple progress line."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and zip_ok(dst):
        print(f"  ok (already here): {dst.relative_to(ROOT)}")
        return
    part = dst.with_suffix(dst.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:                 # server ignored the resume request: start again
            have = 0
        total = have + int(r.headers.get("Content-Length", 0))
        with open(part, "ab" if have else "wb") as f:
            done = have
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dst.name}: {done / 2**30:.2f} / {total / 2**30:.2f} GB", end="", flush=True)
    print()
    part.replace(dst)
    if not zip_ok(dst):
        dst.unlink()
        raise RuntimeError(f"{dst.name} is damaged, run the script again")


def unpack_davis():
    out = DS / "davis" / "DAVIS"
    if (out / "ImageSets" / "2017" / "val.txt").exists():
        return
    z = zipfile.ZipFile(DS / "davis" / "DAVIS-2017-trainval-480p.zip")
    val = z.read("DAVIS/ImageSets/2017/val.txt").decode().split()
    want = [n for n in z.namelist() if n.startswith("DAVIS/ImageSets/") or any(f"/480p/{s}/" in n for s in val)]
    z.extractall(DS / "davis", members=want)
    print(f"  unpacked DAVIS 2017 val: {len(val)} videos")


def get_youtubematte():
    folder = DS / "youtubematte"
    folder.mkdir(parents=True, exist_ok=True)
    import gdown

    for name, fid in YTM_IDS.items():
        dst = folder / name
        if dst.exists() and zip_ok(dst):
            print(f"  ok (already here): {dst.relative_to(ROOT)}")
            continue
        try:
            gdown.download(id=fid, output=str(dst), quiet=False)
        except Exception as e:
            raise RuntimeError(
                f"Google Drive refused {name} ({str(e).splitlines()[0][:80]}). Open "
                f"https://drive.google.com/uc?id={fid} in a browser, save it as {dst}, then run this again.")
    if not (folder / "YouTubeMatte" / "youtubematte_512x288").exists():
        z = zipfile.ZipFile(folder / "YouTubeMatte.zip")
        z.extractall(folder, members=[n for n in z.namelist()
                                      if "youtubematte_512x288/" in n and ("/har/" in n or "/pha/" in n)])
        m = zipfile.ZipFile(folder / "first_frame_mask.zip")
        m.extractall(folder, members=[n for n in m.namelist() if "512x288" in n])
        print("  unpacked YouTubeMatte 512x288")


def main(sets):
    for s in sets:
        print(f"== {s}")
        if s == "youtubematte":
            get_youtubematte()
            continue
        if s not in FILES:
            raise SystemExit(f"unknown set {s!r}; choose from bonn, davis, sintel, youtubematte")
        for url, rel in FILES[s]:
            download(url, DS / rel)
        if s == "davis":
            unpack_davis()


if __name__ == "__main__":
    main(sys.argv[1:] or ["bonn", "davis"])
