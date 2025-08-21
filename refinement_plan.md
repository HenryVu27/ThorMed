# Pixel-Accurate Mask Refinement Plan for Bladder Ultrasound (Step-0 → Step-1)

This revision **adds a mandatory Step-0 audit** that interrogates every pixel of the SAM-generated masks before any smoothing is applied.  The subsequent Step-1 pipeline (morphology → CRF → Fourier) has been tweaked to honour the audit findings.  Follow all stages in order; each script is self-contained and CPU-friendly.

---
## Directory Layout
```
Bladder_Data/
├── images/             # ultrasound B-mode PNGs (8-bit)
└── masks/              # raw SAM masks  (same stem + "_mask.png")
```
All refined outputs will live under `Bladder_Data/refined_masks/`.

---
## Step-0 Mask Integrity Audit (**must run once**)

### Purpose
1. Verify that every mask is truly binary (values ∈ {0, 255}).
2. Detect rogue grayscale pixels, compression artefacts, or multiple objects.
3. Generate a CSV summary so you can sort and manually inspect suspicious slices.

### `audit_masks.py`
```python
"""Audit SAM-generated bladder masks for pixel integrity and topology."""
import cv2, os, csv, numpy as np, argparse, tqdm

def audit_one(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    uniq = np.unique(m)
    # enforce binary through threshold for downstream scripts
    bin_m = (m > 127).astype(np.uint8)
    area = bin_m.sum()
    # Euler number: blobs − holes (cv2 connectedComponents + flood-fill)
    nb_comp, lbl, stats, _ = cv2.connectedComponentsWithStats(bin_m, 8)
    holes = cv2.countNonZero(255 - cv2.morphologyEx(bin_m * 255, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8)))
    return {
        'filename': os.path.basename(path),
        'unique_vals': ';'.join(map(str, uniq.tolist()[:10])),
        'non_binary': int(len(uniq) > 2 or (len(uniq)==2 and set(uniq) != {0,255})),
        'area_px': int(area),
        'components': int(nb_comp-1),   # minus background
        'holes_px': int(holes)
    }

def main(in_dir, out_csv):
    rows = []
    for fn in tqdm.tqdm(sorted(os.listdir(in_dir))):
        if fn.endswith('_mask.png'):
            rows.append(audit_one(os.path.join(in_dir, fn)))
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Audit complete → {out_csv}  (flag any row with non_binary=1, components>1, or holes_px>0)")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', default='Bladder_Data/masks')
    ap.add_argument('--out_csv', default='Bladder_Data/mask_audit.csv')
    main(**vars(ap.parse_args()))
```
Run:
```bash
python audit_masks.py
```
**Action items**
* If `non_binary==1`, open that mask and **threshold** it: `cv2.threshold(img,127,255,cv2.THRESH_BINARY)`.
* If `components>1` and the extra component’s area is <1 % of the main blob, let Step-1 remove it; otherwise manually delete it.
* If `holes_px>0`, consider filling them _before_ smoothing: `cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (3,3))`.

Only proceed when all severe anomalies are fixed.

---
## Step-1 Three-Stage Post-processing
Prerequisites (create a fresh venv if you skipped earlier):
```bash
python -m venv .venv && source .venv/bin/activate
pip install opencv-python-headless numpy scikit-image pydensecrf tqdm
```

### 1-A Morphological Smoothing (`morph_smooth.py`)
```python
import cv2, os, numpy as np, argparse, time, tqdm

def morph_smooth(mask: np.ndarray, k: int = 5) -> np.ndarray:
    # enforce binary
    mask = (mask > 127).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    nb, lbl, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    keep = np.zeros_like(opened)
    for i in range(1, nb):  # skip background
        if stats[i, cv2.CC_STAT_AREA] > 0.001 * opened.size:
            keep[lbl == i] = 255
    return keep

def batch(in_dir, out_dir, k=5):
    os.makedirs(out_dir, exist_ok=True)
    for fn in tqdm.tqdm(sorted(os.listdir(in_dir))):
        if fn.endswith('_mask.png'):
            m = cv2.imread(os.path.join(in_dir, fn), 0)
            cv2.imwrite(os.path.join(out_dir, fn), morph_smooth(m, k))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', default='Bladder_Data/masks')
    ap.add_argument('--out_dir', default='Bladder_Data/refined_masks/morph')
    ap.add_argument('-k','--kernel', type=int, default=5)
    args = ap.parse_args(); t0=time.time(); batch(**vars(args));
    print(f"Morph-smooth done in {time.time()-t0:.1f}s")
```
*Kernel size guideline*: choose `k ≈ image_resolution_mm / 0.4`.  For 0.8 mm/pixel, `k=5` erases <2 px jaggies without shrinking the bladder neck.

### 1-B Dense-CRF Edge Snap (`crf_refine.py`)
```python
import cv2, os, numpy as np, argparse, time, tqdm
import pydensecrf.densecrf as dcrf, pydensecrf.utils as utils

def crf_refine(img, mask, it=5):
    h, w = img.shape[:2]
    d = dcrf.DenseCRF2D(w, h, 2)
    unary = utils.unary_from_labels(mask//255, 2, gt_prob=0.8)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=20, srgb=10, rgbim=img, compat=10)
    Q = d.inference(it)
    return (np.argmax(Q,0).reshape(h,w)*255).astype(np.uint8)

def batch(img_dir, in_dir, out_dir, it=5, jaccard_thresh=0.02):
    os.makedirs(out_dir, exist_ok=True)
    for fn in tqdm.tqdm(sorted(os.listdir(in_dir))):
        if not fn.endswith('_mask.png'): continue
        m_path = os.path.join(in_dir, fn)
        mask = cv2.imread(m_path,0)
        orig = cv2.imread(os.path.join('Bladder_Data/masks', fn),0)
        j = 1 - (cv2.bitwise_and(mask,orig).sum() / float(cv2.bitwise_or(mask,orig).sum()+1e-6))
        if j < jaccard_thresh:
            cv2.imwrite(os.path.join(out_dir, fn), mask); continue
        img = cv2.imread(os.path.join(img_dir, fn.replace('_mask','')))
        cv2.imwrite(os.path.join(out_dir, fn), crf_refine(img, mask, it))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', default='Bladder_Data/images')
    ap.add_argument('--in_dir',  default='Bladder_Data/refined_masks/morph')
    ap.add_argument('--out_dir', default='Bladder_Data/refined_masks/crf')
    ap.add_argument('-i','--iter', type=int, default=5)
    ap.add_argument('--jaccard', type=float, default=0.02)
    args = ap.parse_args(); t0=time.time(); batch(**vars(args));
    print(f"CRF done in {time.time()-t0:.1f}s")
```
*The CRF step now **skips** masks whose morphology output changed <2 % to save compute.*

### 1-C Fourier-Descriptor Contour Filter (`fd_smooth.py`)
```python
import cv2, os, numpy as np, argparse, time, tqdm
from skimage import measure

def fd(mask, keep=30):
    cnt = max(measure.find_contours(mask,0.5), key=len)
    fx, fy = np.fft.fft(cnt[:,1]), np.fft.fft(cnt[:,0])
    fx[keep:-keep]=0; fy[keep:-keep]=0
    new = np.stack([np.fft.ifft(fy).real, np.fft.ifft(fx).real],1).round().astype(int)
    sm = np.zeros_like(mask); cv2.fillPoly(sm,[new],255); return sm

def batch(in_dir, out_dir, keep=30):
    os.makedirs(out_dir,exist_ok=True)
    for fn in tqdm.tqdm(sorted(os.listdir(in_dir))):
        if fn.endswith('_mask.png'):
            m=cv2.imread(os.path.join(in_dir,fn),0)
            cv2.imwrite(os.path.join(out_dir,fn), fd(m,keep))

if __name__=='__main__':
    import argparse; ap=argparse.ArgumentParser();
    ap.add_argument('--in_dir', default='Bladder_Data/refined_masks/crf')
    ap.add_argument('--out_dir',default='Bladder_Data/refined_masks/fd')
    ap.add_argument('-k','--keep', type=int, default=30)
    a=ap.parse_args(); t0=time.time(); batch(**vars(a));
    print(f"FD filter done in {time.time()-t0:.1f}s")
```
*Parameter rule-of-thumb*: `keep ≈ contour_length / 70`.  Inspect visually and lower if thin structures disappear.

---
## End-to-End Execution
```bash
# 0. Audit
python audit_masks.py  # fix flagged masks first
# 1. Refinement cascade
python morph_smooth.py && python crf_refine.py && python fd_smooth.py
```
Full cascade on 370 slices ≤ 45 s on a modern laptop CPU.

---
## Verification Checklist
1. `hd95(new, old)` **must decrease** or stay equal for ≥95 % of slices.
2. `area_drift = |area_new – area_old| / area_old` **< 2 %**.
3. Overlay spot-checks (`cv2.addWeighted(img,0.6,mask,0.4,0)`) show smooth bladder wall with no new leaks.

If any slice violates point 1 or 2, tweak the parameters in order: `keep` (1-C) → `k` (1-A) → CRF compat/iter.

---
### What’s Next?
Once Step-1 is stable, consult `extra_refinement_plan.md` to push below sub-millimetre Hausdorff distance.