"""Click on a frame to choose SAM 2 prompt points, instead of typing pixel numbers.

    python -m common.pick_points data/my_shot/frames/0001.png

A window opens with the frame:
    left click   = a point ON the object (label 1, green)
    right click  = a point that is NOT the object (label 0, red), e.g. a person next to it
    u            = undo the last point
    q or Enter   = finish

It prints the lines to paste under `segmentation:` in configs/<shot>.yaml,
and saves a picture of your points next to the frame (..._points.jpg).
Tip: 3-5 points in the middle of big areas (face, body, legs) work best.
"""
import argparse
from pathlib import Path

import cv2

MAX_WINDOW = 1400   # big frames are shown smaller; clicks are scaled back to real pixels


def prompts_yaml(points, labels, frame=1) -> str:
    pts = ", ".join(f"[{x}, {y}]" for x, y in points)
    lab = ", ".join(str(v) for v in labels)
    return ("segmentation:\n"
            "  prompts:\n"
            f"    - frame: {frame}\n"
            f"      points: [{pts}]\n"
            f"      labels: [{lab}]\n"
            "      box: null\n")


def draw(img, points, labels, scale=1.0):
    out = img.copy()
    for (x, y), lab in zip(points, labels):
        c = (0, 200, 0) if lab == 1 else (0, 0, 230)
        cv2.circle(out, (round(x * scale), round(y * scale)), 10, c, -1)
        cv2.circle(out, (round(x * scale), round(y * scale)), 10, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_png", help="e.g. data/my_shot/frames/0001.png")
    ap.add_argument("--frame", type=int, default=None, help="frame number for the config (default: from the file name)")
    a = ap.parse_args()
    path = Path(a.frame_png)
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    frame_no = a.frame or (int(path.stem) if path.stem.isdigit() else 1)

    scale = min(1.0, MAX_WINDOW / max(img.shape[:2]))
    view = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    points, labels = [], []

    def on_mouse(event, x, y, flags, param):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            points.append((round(x / scale), round(y / scale)))
            labels.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)

    win = "left = object, right = not object, u = undo, q = done"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, draw(view, points, labels, scale))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 13):
            break
        if key == ord("u") and points:
            points.pop()
            labels.pop()
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    if not points:
        print("No points chosen.")
        return
    preview = path.with_name(f"{path.stem}_points.jpg")
    cv2.imwrite(str(preview), draw(img, points, labels))
    print("\nPaste this into your configs/<shot>.yaml (replace any old segmentation: block):\n")
    print(prompts_yaml(points, labels, frame_no))
    print(f"Picture of your points: {preview}")


if __name__ == "__main__":
    main()
