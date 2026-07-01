"""
segmentation_pipeline.py
=========================
Single entry-point for the prescription LINE SEGMENTATION phase.

Takes ONE raw prescription image and returns the final 128x32 medicine line
crops, ready to feed into the HTR model.

Pipeline (the CURRENT, adopted approach — DeepLab is NOT used):
    raw image
      -> paper detection + flatten (perspective correct)
      -> preprocess (grayscale, adaptive threshold, dilate, invert)
      -> CRAFT text detection
      -> contextual filtering (drop printed headers/seals/footers)
      -> line merging (group words into medicine lines)
      -> perspective crop + resize/pad to 128x32
    => list of line crops (in reading order) + metadata

USAGE (in-memory, for HTR integration):
    from segmentation_pipeline import PrescriptionSegmenter

    seg = PrescriptionSegmenter(
        craft_model_path="models/craft_mlt_25k.pth",
        craft_repo_dir="libs/CRAFT-pytorch",
    )
    result = seg.segment("path/to/prescription.jpg")

    for line in result["lines"]:
        crop = line["image"]        # numpy array, 128x32x3, BGR, uint8
        # feed `crop` straight into your HTR model
"""

import os
import sys
import cv2
import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable


class PrescriptionSegmenter:
    def __init__(self, craft_model_path, craft_repo_dir,
                 device=None,
                 # filtering / merging knobs (tuned defaults from development)
                 top_margin_pct=0.10, bottom_margin_pct=0.08,
                 min_region_area=300, max_width_pct=0.92,
                 seal_density_max=0.78,
                 printed_filter_on=True, printed_row_std_max=0.16,
                 printed_min_height=12,
                 same_line_y_tol=0.7, line_merge_gap=120,
                 paper_min_area_ratio=0.20):

        self.cfg = dict(top_margin_pct=top_margin_pct,
                        bottom_margin_pct=bottom_margin_pct,
                        min_region_area=min_region_area,
                        max_width_pct=max_width_pct,
                        seal_density_max=seal_density_max,
                        printed_filter_on=printed_filter_on,
                        printed_row_std_max=printed_row_std_max,
                        printed_min_height=printed_min_height,
                        same_line_y_tol=same_line_y_tol,
                        line_merge_gap=line_merge_gap,
                        paper_min_area_ratio=paper_min_area_ratio)

        # Device selection
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # ---- CRAFT import (with the legacy compat fix) ----
        import torchvision.models.vgg as vgg
        if not hasattr(vgg, 'model_urls'):
            vgg.model_urls = {'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'}
        if craft_repo_dir not in sys.path:
            sys.path.append(craft_repo_dir)

        from craft import CRAFT
        import craft_utils
        import imgproc
        self._craft_utils = craft_utils
        self._imgproc = imgproc

        # ---- load CRAFT weights ----
        net = CRAFT()
        net.load_state_dict(self._copy_state_dict(torch.load(craft_model_path, map_location='cpu')))
        net = net.to(self.device)
        net.eval()
        self.net = net

        self.craft_params = {'text_threshold': 0.7, 'link_threshold': 0.4,
                             'low_text': 0.4, 'canvas_size': 1280,
                             'mag_ratio': 1.5, 'poly': False}

    # ===================================================================
    # PUBLIC ENTRY POINT
    # ===================================================================
    def segment(self, image_path, save_dir=None):
        """
        Run the full segmentation on one prescription image.

        Args:
            image_path : path to the raw prescription image.
            save_dir   : optional. If given, also writes each crop to disk
                         as <basename>_line_NNN.jpg in this folder.

        Returns: dict
            {
              "source": <image_path>,
              "count":  <int number of medicine lines found>,
              "lines": [
                 {
                   "index": 0,
                   "image": <np.ndarray 32x128x3 BGR uint8>,   # HTR input
                   "bbox":  (x, y, w, h),   # location on the flattened page
                   "filename": "<base>_line_000.jpg"           # if saved
                 },
                 ...
              ]
            }
            Lines are ordered top-to-bottom, left-to-right (reading order).
        """
        raw = cv2.imread(image_path)
        if raw is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # 1. paper detect + flatten
        flat = self._flatten_paper(raw)

        # 2. preprocess (the flattened color image is used for cropping;
        #    grayscale is used for filtering features)
        gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
        page_h, page_w = gray.shape

        # 3. CRAFT detect all text
        flat_rgb = cv2.cvtColor(flat, cv2.COLOR_BGR2RGB)
        raw_boxes = self._craft_boxes(flat_rgb)

        # 4. contextual filter
        kept = [b for b in raw_boxes
                if self._is_medicine_region(b, gray, page_h, page_w)]

        # 5. merge into lines, ordered reading order
        lines = self._merge_into_lines(kept, page_h, page_w)
        lines = sorted(lines, key=lambda p: (p[0][1], p[0][0]))

        # 6. crop + normalize each line
        base = os.path.splitext(os.path.basename(image_path))[0]
        out = []
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        for i, poly in enumerate(lines):
            cropped = self._crop_warp(flat, poly)
            final = self._pad_to_size(cropped, 128, 32)
            if final is None:
                continue
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            bbox = (int(min(xs)), int(min(ys)),
                    int(max(xs) - min(xs)), int(max(ys) - min(ys)))
            fname = f"{base}_line_{i:03d}.jpg"
            if save_dir:
                cv2.imwrite(os.path.join(save_dir, fname), final)
            out.append({"index": i, "image": final, "bbox": bbox, "filename": fname})

        return {"source": image_path, "count": len(out), "lines": out}

    # ===================================================================
    # INTERNALS
    # ===================================================================
    @staticmethod
    def _copy_state_dict(state_dict):
        start = 1 if list(state_dict.keys())[0].startswith("module") else 0
        return OrderedDict({".".join(k.split(".")[start:]): v
                            for k, v in state_dict.items()})

    # ---- paper detection ----
    def _flatten_paper(self, img):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return img
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) < h * w * self.cfg["paper_min_area_ratio"]:
            return img
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        corners = (approx.reshape(4, 2) if len(approx) == 4
                   else cv2.boxPoints(cv2.minAreaRect(largest)))
        corners = self._order_corners(corners)
        warped = self._warp_paper(img, corners)
        return warped if warped is not None else img

    @staticmethod
    def _order_corners(pts):
        pts = np.array(pts, dtype=np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1); d = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
        return rect

    @staticmethod
    def _warp_paper(img, corners):
        (tl, tr, br, bl) = corners
        mw = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        mh = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if mw < 50 or mh < 50:
            return None
        dst = np.array([[0, 0], [mw-1, 0], [mw-1, mh-1], [0, mh-1]], dtype=np.float32)
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(corners, dst), (mw, mh))

    # ---- CRAFT ----
    def _craft_boxes(self, image_rgb):
        p = self.craft_params
        img_resized, target_ratio, _ = self._imgproc.resize_aspect_ratio(
            image_rgb, p['canvas_size'], interpolation=cv2.INTER_LINEAR, mag_ratio=p['mag_ratio'])
        ratio = 1 / target_ratio
        x = self._imgproc.normalizeMeanVariance(img_resized)
        x = Variable(torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)).to(self.device)
        with torch.no_grad():
            y, _ = self.net(x)
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()
        boxes, _ = self._craft_utils.getDetBoxes(
            score_text, score_link, p['text_threshold'], p['link_threshold'], p['low_text'], p['poly'])
        boxes = self._craft_utils.adjustResultCoordinates(boxes, ratio, ratio)
        return [b for b in boxes if b is not None]

    # ---- filtering ----
    @staticmethod
    def _box_rect(box, page_h, page_w):
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
        x2, y2 = min(page_w, int(max(xs))), min(page_h, int(max(ys)))
        return x1, y1, x2, y2

    def _looks_printed(self, roi):
        c = self.cfg
        h, w = roi.shape
        if h < c["printed_min_height"] or w < 10:
            return False
        ink = (roi < 128).astype(np.float32)
        row = ink.sum(axis=1); active = row[row > 0]
        if len(active) < 4:
            return False
        m = active.mean()
        if m < 1e-6:
            return False
        return (active.std() / m) < c["printed_row_std_max"]

    def _is_medicine_region(self, box, gray, page_h, page_w):
        c = self.cfg
        x1, y1, x2, y2 = self._box_rect(box, page_h, page_w)
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            return False
        cy = (y1 + y2) / 2
        if cy < page_h * c["top_margin_pct"] or cy > page_h * (1 - c["bottom_margin_pct"]):
            return False
        if w * h < c["min_region_area"] or w > page_w * c["max_width_pct"]:
            return False
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        density = (roi < 128).sum() / roi.size
        aspect = w / float(h)
        if density > c["seal_density_max"] and 0.6 < aspect < 1.7:
            return False
        if c["printed_filter_on"] and self._looks_printed(roi):
            return False
        return True

    # ---- merging ----
    def _merge_into_lines(self, boxes, page_h, page_w):
        c = self.cfg
        if len(boxes) == 0:
            return []
        rects = []
        for b in boxes:
            x1, y1, x2, y2 = self._box_rect(b, page_h, page_w)
            rects.append([x1, y1, x2 - x1, y2 - y1])
        rects.sort(key=lambda r: r[1])
        lines, cur = [], [rects[0]]
        for r in rects[1:]:
            prev = cur[-1]; avg_h = (prev[3] + r[3]) / 2.0
            if abs((r[1]+r[3]/2) - (prev[1]+prev[3]/2)) < avg_h * c["same_line_y_tol"]:
                cur.append(r)
            else:
                lines.append(cur); cur = [r]
        lines.append(cur)
        merged = []
        for line in lines:
            line.sort(key=lambda r: r[0]); box = line[0]
            for nb in line[1:]:
                if nb[0] - (box[0] + box[2]) < c["line_merge_gap"]:
                    nx, ny = box[0], min(box[1], nb[1])
                    nx2 = max(box[0]+box[2], nb[0]+nb[2]); ny2 = max(box[1]+box[3], nb[1]+nb[3])
                    box = [nx, ny, nx2-nx, ny2-ny]
                else:
                    merged.append(box); box = nb
            merged.append(box)
        return [[[x, y], [x+w, y], [x+w, y+h], [x, y+h]] for x, y, w, h in merged]

    @staticmethod
    def _crop_warp(img, poly):
        pts = np.array(poly, dtype=np.float32)
        mw = int(max(np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[3]-pts[2])))
        mh = int(max(np.linalg.norm(pts[0]-pts[3]), np.linalg.norm(pts[1]-pts[2])))
        if mw == 0 or mh == 0:
            return None
        dst = np.array([[0, 0], [mw-1, 0], [mw-1, mh-1], [0, mh-1]], dtype="float32")
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (mw, mh))

    @staticmethod
    def _pad_to_size(img, tw=128, th=32):
        if img is None or img.size == 0:
            return None
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return None
        s = min(tw/w, th/h)
        nw, nh = max(1, int(w*s)), max(1, int(h*s))
        res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        out = np.ones((th, tw, 3), dtype=np.uint8) * 255
        yo = (th - nh) // 2
        out[yo:yo+nh, 0:nw, :] = res
        return out


if __name__ == "__main__":
    # quick self-test
    seg = PrescriptionSegmenter(
        craft_model_path="models/craft_mlt_25k.pth",
        craft_repo_dir="libs/CRAFT-pytorch",
    )
    res = seg.segment("data/test_images/176820_0.Jpg", save_dir="output/line_crops")
    print(f"Found {res['count']} medicine lines")
    for ln in res["lines"]:
        print(f"  {ln['filename']}  bbox={ln['bbox']}  shape={ln['image'].shape}")
