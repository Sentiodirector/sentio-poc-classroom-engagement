
import cv2
import json
import base64
import numpy as np
from pathlib import Path
from datetime import date
from collections import defaultdict
from PIL import Image
import io

# ---------------------------------------------------------------------------
# CONFIG  (do not rename these)
# ---------------------------------------------------------------------------
VIDEO_PATH      = Path("video_sample_1.mov")
REPORT_HTML_OUT = Path("engagement_report.html")
OUTPUT_JSON     = Path("engagement_output.json")

WINDOW_SEC      = 6       # seconds per analysis window — keep this configurable
SAMPLE_EVERY_N  = 15      # analyse every Nth frame inside a window (speed trade-off) — OPTIMIZED
GRID_ROWS       = 4
GRID_COLS       = 6
FRAME_SCALE     = 0.5     # downscale frames to 50% for faster detection

# ---------------------------------------------------------------------------
# MediaPipe setup (OPTIMIZED: Initialize once at module level)
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp
    _mp_face_mesh      = mp.solutions.face_mesh
    _mp_face_detection = mp.solutions.face_detection
    _MP_AVAILABLE      = True
    # Create detectors ONCE to avoid expensive re-initialization
    _face_detector     = _mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.4
    )
    _face_mesh         = _mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.3,
    )
except ImportError:
    _MP_AVAILABLE = False
    _face_detector = None
    _face_mesh = None
    print("[WARN] mediapipe not found — falling back to Haar cascade")


# ───────────────────────────────────────────────────────────────────────────
# GAZE & EYE OPENNESS
# ───────────────────────────────────────────────────────────────────────────

def estimate_gaze(face_crop: np.ndarray) -> str:
    """
    Return one of: "forward", "down", "left", "right", "unknown"

    Uses MediaPipe Face Mesh iris landmarks (indices 468-477).
    Computes horizontal offset of iris centre from eye centre,
    normalised to eye width. Threshold at +/-0.10 -> left/right.
    Vertical offset > 0.12 -> down.
    Fallback: brightness gradient across the eye strip.
    """
    if face_crop is None or face_crop.size == 0:
        return "unknown"

    h, w = face_crop.shape[:2]
    if h < 20 or w < 20:
        return "unknown"

    # MediaPipe iris path (OPTIMIZED: Use global detector)
    if _MP_AVAILABLE and _face_mesh is not None:
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        res = _face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            def _norm(idx):
                return np.array([lm[idx].x * w, lm[idx].y * h])

            # Left eye: outer=33, inner=133, iris-centre=468
            eye_l_out  = _norm(33)
            eye_l_in   = _norm(133)
            iris_l     = _norm(468)
            eye_width  = np.linalg.norm(eye_l_in - eye_l_out) + 1e-6
            eye_centre = (eye_l_out + eye_l_in) / 2

            upper = _norm(159)
            lower = _norm(145)
            eye_h = np.linalg.norm(upper - lower) + 1e-6

            horiz = (iris_l[0] - eye_centre[0]) / eye_width
            vert  = (iris_l[1] - eye_centre[1]) / eye_h

            if vert > 0.12:
                return "down"
            if horiz < -0.10:
                return "left"
            if horiz > 0.10:
                return "right"
            return "forward"

    # Brightness gradient fallback
    gray      = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    eye_strip = gray[int(h * 0.25): int(h * 0.55), :]
    if eye_strip.size == 0:
        return "unknown"
    left_half  = float(eye_strip[:, :w // 2].mean())
    right_half = float(eye_strip[:, w // 2:].mean())
    diff = right_half - left_half
    if abs(diff) < 8:
        return "forward"
    return "left" if diff > 0 else "right"


def estimate_eye_openness(face_crop: np.ndarray) -> float:
    """
    Eye height / eye width ratio from face mesh, scaled 0-100.
    Return 50.0 if no landmark found.
    """
    if face_crop is None or face_crop.size == 0:
        return 50.0

    h, w = face_crop.shape[:2]
    if not _MP_AVAILABLE or h < 20 or w < 20 or _face_mesh is None:
        return 50.0

    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    res = _face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        return 50.0

    lm = res.multi_face_landmarks[0].landmark

    def _pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    upper   = _pt(159)
    lower   = _pt(145)
    outer   = _pt(33)
    inner   = _pt(133)
    eye_h   = np.linalg.norm(upper - lower)
    eye_w   = np.linalg.norm(outer - inner) + 1e-6
    ratio   = eye_h / eye_w             # typical open ~0.25-0.35
    # scale: ratio 0.30 -> 75, ratio 0.10 -> 25
    openness = min(100.0, max(0.0, ratio * 250))
    return round(openness, 1)


# ───────────────────────────────────────────────────────────────────────────
# PER-FACE ENGAGEMENT SCORE
# ───────────────────────────────────────────────────────────────────────────

def score_face(face_crop: np.ndarray) -> dict:
    """
    Compute engagement for one face.
    Return: { "score": float, "gaze": str, "eye_openness": float }

    Formula from README:
      face_engagement = (forward_gaze x 40) + (eye_openness x 0.25) + (head_pose x 0.35)
      forward_gaze = 1 if gaze == "forward" else 0.25
      head_pose    = 1.0 if gaze == "forward" else 0.3  (simple proxy)
    """
    gaze         = estimate_gaze(face_crop)
    eye_openness = estimate_eye_openness(face_crop)

    forward_gaze = 1.0 if gaze == "forward" else 0.25
    head_pose    = 1.0 if gaze == "forward" else 0.3

    score = (forward_gaze * 40) + (eye_openness * 0.25) + (head_pose * 0.35)
    score = round(min(100.0, max(0.0, score)), 2)

    return {"score": score, "gaze": gaze, "eye_openness": eye_openness}


# ───────────────────────────────────────────────────────────────────────────
# SPATIAL ZONE
# ───────────────────────────────────────────────────────────────────────────

def get_zone(bbox: tuple, frame_w: int, frame_h: int) -> str:
    """
    bbox = (x, y, w, h) in pixels. Use face centre.
    Return zone ID string like "R2C3".
    Clamp column 0-5, row 0-3.
    """
    x, y, bw, bh = bbox
    cx = x + bw // 2
    cy = y + bh // 2

    col = int((cx / frame_w) * GRID_COLS)
    row = int((cy / frame_h) * GRID_ROWS)

    col = min(col, GRID_COLS - 1)
    row = min(row, GRID_ROWS - 1)

    return f"R{row + 1}C{col + 1}"


# ───────────────────────────────────────────────────────────────────────────
# FACE DETECTION IN ONE FRAME
# ───────────────────────────────────────────────────────────────────────────

def detect_faces(frame: np.ndarray) -> list:
    """
    Detect all faces. Return: [{"bbox": (x,y,w,h), "face_crop": ndarray}, ...]
    Apply CLAHE on the frame first.
    Uses MediaPipe Face Detection; falls back to Haar cascade.
    """
    # CLAHE pre-processing
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    frame_eq = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    detections = []
    fh, fw = frame.shape[:2]

    # MediaPipe Face Detection (OPTIMIZED: Use global detector, downscale frame)
    if _MP_AVAILABLE and _face_detector is not None:
        # Downscale frame for faster processing
        small_frame = cv2.resize(frame_eq, (int(fw * FRAME_SCALE), int(fh * FRAME_SCALE)))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        res = _face_detector.process(rgb)

        if res.detections:
            for det in res.detections:
                bb = det.location_data.relative_bounding_box
                # Scale back to original frame dimensions
                x  = max(0, int(bb.xmin * fw))
                y  = max(0, int(bb.ymin * fh))
                w  = min(int(bb.width  * fw), fw - x)
                h  = min(int(bb.height * fh), fh - y)
                if w > 15 and h > 15:
                    crop = frame[y:y+h, x:x+w]
                    detections.append({"bbox": (x, y, w, h), "face_crop": crop})
        return detections

    # Haar cascade fallback
    gray = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2GRAY)
    cc   = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in (faces if len(faces) else []):
        detections.append({"bbox": (x, y, w, h), "face_crop": frame[y:y+h, x:x+w]})
    return detections


# ───────────────────────────────────────────────────────────────────────────
# PROCESS ONE TIME WINDOW
# ───────────────────────────────────────────────────────────────────────────

def process_window(frames_in_window: list, frame_w: int, frame_h: int) -> dict:
    """
    frames_in_window: [(frame_idx, timestamp_sec, ndarray), ...]

    For each sampled frame:
      - detect faces
      - score each face
      - assign each face to a spatial zone

    Aggregate:
      - group engagement_score = mean of all face scores
      - gaze_distribution count
      - spatial_zones = mean score per zone

    Return a dict that becomes one entry in time_windows.
    Also attach face_crops list (a few small crops for evidence thumbnails - not in JSON).
    """
    gaze_dist  = {"forward": 0, "down": 0, "left": 0, "right": 0, "unknown": 0}
    zone_accum = defaultdict(list)
    all_scores = []
    all_crops  = []

    for _, _, frame in frames_in_window:
        faces = detect_faces(frame)
        for f in faces:
            result = score_face(f["face_crop"])
            zone   = get_zone(f["bbox"], frame_w, frame_h)

            all_scores.append(result["score"])
            gaze_dist[result["gaze"]] = gaze_dist.get(result["gaze"], 0) + 1
            zone_accum[zone].append(result["score"])
            all_crops.append(f["face_crop"])

    group_score   = round(float(np.mean(all_scores)), 2) if all_scores else 0
    spatial_zones = {z: round(float(np.mean(v)), 2) for z, v in zone_accum.items()}

    return {
        "engagement_score":  group_score,
        "persons_count":     len(all_scores),
        "gaze_distribution": gaze_dist,
        "spatial_zones":     spatial_zones,
        "face_crops":        all_crops,   # internal only, stripped before JSON
    }


# ───────────────────────────────────────────────────────────────────────────
# COLLAGE HELPER
# ───────────────────────────────────────────────────────────────────────────

def make_collage_b64(face_crops: list, cell: int = 60) -> str:
    """
    Stack face crops horizontally into a small collage, return as base64 JPEG.
    Resize each crop to cell x cell before stacking.
    Return empty string if no crops.
    """
    if not face_crops:
        return ""

    cells = []
    for crop in face_crops[:10]:
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((cell, cell), Image.LANCZOS)
            cells.append(np.array(img))
        except Exception:
            pass

    if not cells:
        return ""

    collage = np.hstack(cells)
    buf     = io.BytesIO()
    Image.fromarray(collage).save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


# ───────────────────────────────────────────────────────────────────────────
# COLOUR HELPER
# ───────────────────────────────────────────────────────────────────────────

def _score_to_hex(score: float) -> str:
    """0 -> red #ef4444   50 -> white #ffffff   100 -> green #22c55e"""
    s = max(0.0, min(100.0, float(score)))
    if s <= 50:
        t = s / 50.0
        r = int(239 + (255 - 239) * t)
        g = int(68  + (255 - 68)  * t)
        b = int(68  + (255 - 68)  * t)
    else:
        t = (s - 50) / 50.0
        r = int(255 - (255 - 34)  * t)
        g = int(255 - (255 - 197) * t)
        b = int(255 - (255 - 94)  * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _text_colour(score: float) -> str:
    return "#1a1a1a" if 20 < score < 80 else "#ffffff"


# ───────────────────────────────────────────────────────────────────────────
# HTML REPORT
# ───────────────────────────────────────────────────────────────────────────

def generate_engagement_report(windows: list, stats: dict, output_path: Path):
    """
    Write engagement_report.html - self-contained, no CDN.

    Must include:
      1. Session summary numbers
      2. Engagement timeline chart  (pure canvas - no CDN)
      3. 4x6 heatmap: HTML table, cell background-color based on zone score
         colour scale: red (#ef4444) at 0, white at 50, green (#22c55e) at 100
      4. 3 worst windows: timestamp + score + face collage image
    """
    labels = [f"{w['start_sec']:.0f}s" for w in windows]
    scores = [w["engagement_score"] for w in windows]

    # Heatmap rows
    heatmap = stats["session_spatial_heatmap"]
    heatmap_rows = ""
    for r in range(GRID_ROWS):
        heatmap_rows += "<tr>"
        for c in range(GRID_COLS):
            zid   = f"R{r+1}C{c+1}"
            score = heatmap.get(zid, 0)
            bg    = _score_to_hex(score)
            fg    = _text_colour(score)
            heatmap_rows += (
                f'<td style="background:{bg};color:{fg};'
                f'padding:10px 6px;text-align:center;font-size:11px;'
                f'border:1px solid #e5e7eb;">'
                f'<span style="font-weight:600">{zid}</span><br>'
                f'<span style="font-size:13px">{score}</span></td>'
            )
        heatmap_rows += "</tr>"

    # Worst 3 cards
    worst_cards = ""
    for i, w in enumerate(stats["worst_3_windows"]):
        b64 = w.get("thumbnail_collage_b64", "")
        img_tag = (
            f'<img src="data:image/jpeg;base64,{b64}" '
            f'style="width:100%;border-radius:6px;margin-top:8px;" alt="faces"/>'
            if b64 else
            '<p style="color:#9ca3af;font-size:12px;margin-top:8px;">No faces detected</p>'
        )
        score_val = w["score"]
        badge_col = "#ef4444" if score_val < 50 else "#f59e0b"
        worst_cards += f"""
        <div style="flex:1;min-width:200px;background:#fff;border:1px solid #e5e7eb;
                    border-radius:10px;padding:14px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-weight:600;color:#374151;">Worst #{i+1}</span>
            <span style="background:{badge_col};color:#fff;border-radius:4px;
                         padding:2px 8px;font-size:12px;">{score_val:.1f}%</span>
          </div>
          <div style="font-size:12px;color:#6b7280;margin-top:6px;">
            Window {w['window_id']} &nbsp;|&nbsp;
            {w['start_sec']:.1f}s - {w['end_sec']:.1f}s
          </div>
          {img_tag}
        </div>"""

    overall  = stats["overall_engagement_score"]
    ov_color = "#22c55e" if overall >= 70 else ("#f59e0b" if overall >= 50 else "#ef4444")
    peak     = stats["peak_window"]
    trough   = stats["trough_window"]

    js_labels = json.dumps(labels)
    js_scores = json.dumps(scores)
    point_colors = json.dumps([
        "#22c55e" if s >= 70 else ("#f59e0b" if s >= 50 else "#ef4444")
        for s in scores
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Classroom Engagement Report - Sentio Mind</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:system-ui,-apple-system,sans-serif;background:#f3f4f6;
        color:#111827;padding:24px;}}
  h1{{font-size:22px;font-weight:700;color:#1e3a5f;margin-bottom:4px;}}
  h2{{font-size:16px;font-weight:600;color:#374151;margin:24px 0 12px;}}
  .card{{background:#fff;border-radius:12px;padding:20px;
          box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:20px;}}
  .badge{{display:inline-flex;align-items:center;gap:6px;background:#f9fafb;
           border:1px solid #e5e7eb;border-radius:8px;padding:10px 16px;font-size:13px;}}
  .badge strong{{font-size:22px;color:#111827;}}
  .badges{{display:flex;flex-wrap:wrap;gap:12px;}}
  table{{border-collapse:collapse;width:100%;}}
  .thumbs{{display:flex;flex-wrap:wrap;gap:14px;}}
  canvas{{display:block;max-width:100%;}}
  footer{{text-align:center;color:#9ca3af;font-size:12px;margin-top:32px;}}
</style>
</head>
<body>
<div class="card">
  <h1>Classroom Engagement Report</h1>
  <p style="color:#6b7280;font-size:13px;margin-top:4px;">
    Sentio Mind &nbsp;·&nbsp; {stats['video']} &nbsp;·&nbsp; {stats['date']}
  </p>
</div>

<div class="card">
  <h2 style="margin-top:0">Session Summary</h2>
  <div class="badges">
    <div class="badge"><div>
      <strong style="color:{ov_color}">{overall}%</strong><br>
      <span>Overall engagement</span>
    </div></div>
    <div class="badge"><div>
      <strong>{stats['session_duration_sec']:.0f}s</strong><br>
      <span>Session duration</span>
    </div></div>
    <div class="badge"><div>
      <strong>{stats['total_persons_detected']}</strong><br>
      <span>Person detections</span>
    </div></div>
    <div class="badge"><div>
      <strong style="color:#22c55e">{peak['score']}%</strong><br>
      <span>Peak window {peak['window_id']} @ {peak['start_sec']:.0f}s</span>
    </div></div>
    <div class="badge"><div>
      <strong style="color:#ef4444">{trough['score']}%</strong><br>
      <span>Trough window {trough['window_id']} @ {trough['start_sec']:.0f}s</span>
    </div></div>
  </div>
</div>

<div class="card">
  <h2 style="margin-top:0">Group Engagement Timeline</h2>
  <p style="font-size:12px;color:#6b7280;margin-bottom:12px;">
    One point per {WINDOW_SEC}-second window.
    <span style="color:#22c55e">&#9679;</span> >=70 engaged &nbsp;
    <span style="color:#f59e0b">&#9679;</span> 50-70 moderate &nbsp;
    <span style="color:#ef4444">&#9679;</span> &lt;50 low
  </p>
  <canvas id="chart" height="80"></canvas>
</div>

<div class="card">
  <h2 style="margin-top:0">Spatial Heatmap (4 x 6 Seat Grid)</h2>
  <p style="font-size:12px;color:#6b7280;margin-bottom:12px;">
    Row 1 = front of room | Column 1 = leftmost (camera view).
    Colour: <span style="color:#ef4444">red=0</span> -> white=50 ->
    <span style="color:#22c55e">green=100</span>
  </p>
  <div style="overflow-x:auto;">
    <table>
      <thead><tr style="background:#f9fafb;">
        {''.join(f'<th style="padding:6px 12px;font-size:12px;color:#6b7280;border:1px solid #e5e7eb;">Col {c+1}</th>' for c in range(GRID_COLS))}
      </tr></thead>
      <tbody>{heatmap_rows}</tbody>
    </table>
  </div>
</div>

<div class="card">
  <h2 style="margin-top:0">3 Worst Engagement Windows</h2>
  <p style="font-size:12px;color:#6b7280;margin-bottom:12px;">
    Flagged for counsellor review. Face thumbnails from sampled frames.
  </p>
  <div class="thumbs">{worst_cards}</div>
</div>

<footer>Sentio Mind · Classroom Engagement Analysis · {stats['date']}</footer>

<script>
(function(){{
  const labels = {js_labels};
  const scores = {js_scores};
  const dotCol = {point_colors};
  const canvas = document.getElementById('chart');
  const ctx    = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth - 40 || 760;
  canvas.height= 220;
  const W=canvas.width,H=canvas.height;
  const P={{l:44,r:16,t:16,b:44}};
  const cW=W-P.l-P.r,cH=H-P.t-P.b;
  const n=scores.length;

  ctx.fillStyle='#f9fafb'; ctx.fillRect(0,0,W,H);

  [0,25,50,75,100].forEach(v=>{{
    const y=P.t+cH*(1-v/100);
    ctx.strokeStyle='#e5e7eb';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(P.l,y);ctx.lineTo(P.l+cW,y);ctx.stroke();
    ctx.fillStyle='#9ca3af';ctx.font='11px system-ui';ctx.textAlign='right';
    ctx.fillText(v+'%',P.l-6,y+4);
  }});

  const bands=[
    {{y0:0.7,y1:1.0,col:'rgba(34,197,94,.06)'}},
    {{y0:0.5,y1:0.7,col:'rgba(245,158,11,.06)'}},
    {{y0:0.0,y1:0.5,col:'rgba(239,68,68,.06)'}},
  ];
  bands.forEach(b=>{{
    ctx.fillStyle=b.col;
    ctx.fillRect(P.l,P.t+cH*(1-b.y1),cW,cH*(b.y1-b.y0));
  }});

  ctx.beginPath();ctx.strokeStyle='#1e3a5f';ctx.lineWidth=2.5;
  ctx.lineJoin='round';ctx.lineCap='round';
  scores.forEach((v,i)=>{{
    const x=P.l+(n>1?i/(n-1):0.5)*cW;
    const y=P.t+cH*(1-v/100);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }});
  ctx.stroke();

  const step=Math.max(1,Math.floor(n/12));
  scores.forEach((v,i)=>{{
    const x=P.l+(n>1?i/(n-1):0.5)*cW;
    const y=P.t+cH*(1-v/100);
    ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);
    ctx.fillStyle=dotCol[i];ctx.fill();
    ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
    if(i%step===0){{
      ctx.fillStyle='#9ca3af';ctx.font='10px system-ui';
      ctx.textAlign='center';ctx.fillText(labels[i],x,H-P.b+16);
    }}
  }});

  ctx.fillStyle='#6b7280';ctx.font='11px system-ui';ctx.textAlign='center';
  ctx.fillText('Time',P.l+cW/2,H-4);
}})();
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"[report] wrote -> {output_path}")


# ───────────────────────────────────────────────────────────────────────────
# MAIN  (exact same structure as template - only process_window etc. filled in)
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cap   = cv2.VideoCapture(str(VIDEO_PATH))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur   = total / fps

    frames_per_window = max(1, int(WINDOW_SEC * fps))
    windows     = []
    window_buf  = []
    w_start_sec = 0.0
    frame_idx   = 0

    print(f"Processing {VIDEO_PATH}  |  {dur:.1f}s  |  {total} frames  |  window={WINDOW_SEC}s")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps

        if frame_idx % SAMPLE_EVERY_N == 0:
            window_buf.append((frame_idx, ts, frame))  # Skip copy for speed

        if (frame_idx + 1) % frames_per_window == 0 or frame_idx == total - 1:
            result = process_window(window_buf, fw, fh)
            result.update({
                "window_id": len(windows) + 1,
                "start_sec": round(w_start_sec, 2),
                "end_sec":   round(ts, 2),
            })
            windows.append(result)
            window_buf  = []
            w_start_sec = ts + 1 / fps

        frame_idx += 1
    cap.release()

    # Session heatmap
    zone_scores = defaultdict(list)
    for w in windows:
        for zid, score in w["spatial_zones"].items():
            zone_scores[zid].append(score)
    heatmap = {
        f"R{r+1}C{c+1}": int(np.mean(zone_scores[f"R{r+1}C{c+1}"])) if zone_scores[f"R{r+1}C{c+1}"] else 0
        for r in range(GRID_ROWS) for c in range(GRID_COLS)
    }

    scores   = [w["engagement_score"] for w in windows]
    peak     = max(windows, key=lambda w: w["engagement_score"])
    trough   = min(windows, key=lambda w: w["engagement_score"])
    worst3   = sorted(windows, key=lambda w: w["engagement_score"])[:3]

    for w in worst3:
        w["thumbnail_collage_b64"] = make_collage_b64(w.get("face_crops", []))

    def clean(w):
        return {k: v for k, v in w.items() if k != "face_crops"}

    def dominant_gaze(w):
        gd = w.get("gaze_distribution", {})
        return max(gd, key=gd.get) if gd else "unknown"

    stats = {
        "source":                   "p3_classroom_engagement",
        "video":                    str(VIDEO_PATH),
        "date":                     str(date.today()),
        "session_duration_sec":     round(dur, 2),
        "window_size_sec":          WINDOW_SEC,
        "total_persons_detected":   sum(w["persons_count"] for w in windows),
        "overall_engagement_score": int(np.mean(scores)) if scores else 0,
        "peak_window": {
            "window_id":     peak["window_id"],
            "start_sec":     peak["start_sec"],
            "end_sec":       peak["end_sec"],
            "score":         peak["engagement_score"],
            "dominant_gaze": dominant_gaze(peak),
        },
        "trough_window": {
            "window_id":     trough["window_id"],
            "start_sec":     trough["start_sec"],
            "end_sec":       trough["end_sec"],
            "score":         trough["engagement_score"],
            "dominant_gaze": dominant_gaze(trough),
        },
        "worst_3_windows": [
            {
                "window_id":             w["window_id"],
                "start_sec":             w["start_sec"],
                "end_sec":               w["end_sec"],
                "score":                 w["engagement_score"],
                "thumbnail_collage_b64": w.get("thumbnail_collage_b64", ""),
            }
            for w in worst3
        ],
        "time_windows":            [clean(w) for w in windows],
        "session_spatial_heatmap": heatmap,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[json]   wrote -> {OUTPUT_JSON}")

    generate_engagement_report(windows, stats, REPORT_HTML_OUT)

    print()
    print("=" * 50)
    print(f"  Overall engagement:  {stats['overall_engagement_score']}%")
    print(f"  Peak:    window {peak['window_id']}  @ {peak['start_sec']:.0f}s  =  {peak['engagement_score']}%")
    print(f"  Trough:  window {trough['window_id']}  @ {trough['start_sec']:.0f}s  =  {trough['engagement_score']}%")
    print(f"  Report -> {REPORT_HTML_OUT}")
    print(f"  JSON   -> {OUTPUT_JSON}")
    print("=" * 50)