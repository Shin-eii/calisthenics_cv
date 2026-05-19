import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_pose = mp.solutions.pose

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

DISPLAY_W, DISPLAY_H = 960, 540   # standard display resolution


def _fit_to_display(frame):
    """
    Resize frame to fit inside DISPLAY_W x DISPLAY_H while preserving
    aspect ratio. Adds black bars on the shorter axis (letterbox / pillarbox).
    """
    fh, fw = frame.shape[:2]
    scale  = min(DISPLAY_W / fw, DISPLAY_H / fh)
    new_w  = int(fw * scale)
    new_h  = int(fh * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Centre on a black canvas
    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    x_off  = (DISPLAY_W - new_w) // 2
    y_off  = (DISPLAY_H - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _make_pose(is_image: bool):
    return mp_pose.Pose(
        static_image_mode=is_image,
        model_complexity=2 if is_image else 1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _check_frog_stand(landmarks) -> tuple[bool, list]:
    """
    Scoring-based frog stand checker.

    Returns (is_correct, rows) where each row is a dict:
      {
        "name"   : str,           signal label
        "status" : "pass"|"fail"|"skip",
        "detail" : str,           measured value vs threshold
      }

    "skip" means the required landmarks were too low-visibility to measure.
    The pose is CORRECT when ≥ 75% of non-skipped signals pass.
    """
    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    le = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    re = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    rows = []

    # ── Wrist visibility check (signal 1) ─────────────────────────────────────
    wrist_vis = min(lw.visibility, rw.visibility)
    wrist_y   = (lw.y + rw.y) / 2
    if wrist_vis < 0.3:
        rows.append({
            "name":   "1 Wrists low",
            "status": "skip",
            "detail": f"vis={wrist_vis:.2f} < 0.30"
        })
    else:
        ok = wrist_y > 0.55
        rows.append({
            "name":   "1 Wrists low",
            "status": "pass" if ok else "fail",
            "detail": f"y={wrist_y:.2f} {'>' if ok else '<='} 0.55"
        })

    # ── Signal 2: Torso horizontal ─────────────────────────────────────────────
    shoulder_vis = (ls.visibility + rs.visibility) / 2
    shoulder_y   = (ls.y + rs.y) / 2
    if shoulder_vis < 0.3:
        rows.append({
            "name":   "2 Torso horiz",
            "status": "skip",
            "detail": f"shoulder vis={shoulder_vis:.2f} < 0.30"
        })
    else:
        gap = abs(shoulder_y - wrist_y)
        ok  = gap < 0.30
        rows.append({
            "name":   "2 Torso horiz",
            "status": "pass" if ok else "fail",
            "detail": f"|sh-wr|={gap:.2f} {'<' if ok else '>='} 0.30"
        })

    # ── Signal 3: Hips above shoulders ────────────────────────────────────────
    hip_vis = (lh.visibility + rh.visibility) / 2
    hip_y   = (lh.y + rh.y) / 2
    if shoulder_vis < 0.3 or hip_vis < 0.3:
        rows.append({
            "name":   "3 Hips > shoulders",
            "status": "skip",
            "detail": f"hip vis={hip_vis:.2f}"
        })
    else:
        ok = hip_y < shoulder_y
        rows.append({
            "name":   "3 Hips > shoulders",
            "status": "pass" if ok else "fail",
            "detail": f"hip_y={hip_y:.2f} {'<' if ok else '>='} sh_y={shoulder_y:.2f}"
        })

    # ── Signal 4: Elbows bent ─────────────────────────────────────────────────
    l_arm_vis = min(ls.visibility, le.visibility, lw.visibility)
    r_arm_vis = min(rs.visibility, re.visibility, rw.visibility)
    if l_arm_vis < 0.3 and r_arm_vis < 0.3:
        rows.append({
            "name":   "4 Elbows bent",
            "status": "skip",
            "detail": f"arm vis L={l_arm_vis:.2f} R={r_arm_vis:.2f}"
        })
    else:
        l_angle = _calculate_angle(ls, le, lw) if l_arm_vis >= 0.3 else 180
        r_angle = _calculate_angle(rs, re, rw) if r_arm_vis >= 0.3 else 180
        ok = l_angle < 155 or r_angle < 155
        rows.append({
            "name":   "4 Elbows bent",
            "status": "pass" if ok else "fail",
            "detail": f"L={l_angle:.0f} R={r_angle:.0f} (need <155)"
        })

    # ── Signal 5: Feet / knees lifted ─────────────────────────────────────────
    left_up = right_up = False
    left_src = right_src = "—"
    any_vis  = False

    if la.visibility > 0.1:
        any_vis  = True
        left_up  = (wrist_y - la.y) > 0.10
        left_src = f"ankle_y={la.y:.2f}"
    elif lk.visibility > 0.3:
        any_vis  = True
        left_up  = (wrist_y - lk.y) > 0.05
        left_src = f"knee_y={lk.y:.2f}"

    if ra.visibility > 0.1:
        any_vis   = True
        right_up  = (wrist_y - ra.y) > 0.10
        right_src = f"ankle_y={ra.y:.2f}"
    elif rk.visibility > 0.3:
        any_vis   = True
        right_up  = (wrist_y - rk.y) > 0.05
        right_src = f"knee_y={rk.y:.2f}"

    if not any_vis:
        rows.append({
            "name":   "5 Lower body up",
            "status": "skip",
            "detail": "no ankle/knee visible"
        })
    else:
        ok = left_up or right_up
        rows.append({
            "name":   "5 Lower body up",
            "status": "pass" if ok else "fail",
            "detail": f"L:{left_src} R:{right_src}"
        })

    # ── Score ─────────────────────────────────────────────────────────────────
    checked = [r for r in rows if r["status"] != "skip"]
    passed  = [r for r in checked if r["status"] == "pass"]
    score   = len(passed)
    total   = len(checked)
    is_correct = total > 0 and (score / total) >= 0.75

    return is_correct, rows


def _annotate_frame(frame, landmarks, is_correct: bool, rows: list):
    h, w, _ = frame.shape
    skel_color = (0, 255, 0) if is_correct else (0, 0, 255)

    # Draw skeleton
    for connection in mp_pose.POSE_CONNECTIONS:
        s = landmarks[connection[0]]
        e = landmarks[connection[1]]
        if s.visibility > 0.3 and e.visibility > 0.3:
            cv2.line(frame,
                     (int(s.x * w), int(s.y * h)),
                     (int(e.x * w), int(e.y * h)),
                     skel_color, 2)

    for lm in landmarks:
        if lm.visibility > 0.3:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, skel_color, -1)

    # ── Signal panel ──────────────────────────────────────────────────────────
    PANEL_W    = 330
    ROW_H      = 22
    HEADER_H   = 36
    panel_h    = HEADER_H + ROW_H * len(rows) + 8
    cv2.rectangle(frame, (8, 8), (8 + PANEL_W, 8 + panel_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (8, 8), (8 + PANEL_W, 8 + panel_h), (80, 80, 80), 1)

    # Header
    checked = [r for r in rows if r["status"] != "skip"]
    passed  = len([r for r in checked if r["status"] == "pass"])
    total   = len(checked)
    main_label  = "CORRECT FORM" if is_correct else "INCORRECT FORM"
    score_label = f"  ({passed}/{total})"
    cv2.putText(frame, main_label + score_label,
                (14, 8 + HEADER_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, skel_color, 2)

    # Signal rows
    STATUS_COLOR = {
        "pass": (0, 220, 0),    # green
        "fail": (0, 80, 255),   # red
        "skip": (140, 140, 140) # grey
    }
    STATUS_ICON = {"pass": "OK", "fail": " X", "skip": "--"}

    y = 8 + HEADER_H + ROW_H - 6
    for row in rows:
        color = STATUS_COLOR[row["status"]]
        icon  = STATUS_ICON[row["status"]]
        text  = f" {icon}  {row['name']:<20} {row['detail']}"
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        y += ROW_H

    return frame


def _draw_timer(frame, hold_start, best_hold):
    """
    Draw a timer panel in the bottom-left corner.

    hold_start : float | None  — time.time() when current hold began, None if resting
    best_hold  : float         — longest hold so far in seconds
    """
    now          = time.time()
    is_holding   = hold_start is not None
    current_secs = (now - hold_start) if is_holding else 0.0

    h, w, _ = frame.shape

    # Panel dimensions
    PANEL_W = 240
    PANEL_H = 80
    px, py  = 8, h - PANEL_H - 8          # bottom-left

    cv2.rectangle(frame, (px, py), (px + PANEL_W, py + PANEL_H), (20, 20, 20), -1)
    cv2.rectangle(frame, (px, py), (px + PANEL_W, py + PANEL_H), (80, 80, 80), 1)

    # State label
    if is_holding:
        state_text  = "HOLDING"
        state_color = (0, 220, 0)
    else:
        state_text  = "REST"
        state_color = (140, 140, 140)

    cv2.putText(frame, state_text,
                (px + 8, py + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)

    # Current hold — large clock-style
    cur_str = f"{current_secs:05.2f}s"
    cv2.putText(frame, cur_str,
                (px + 8, py + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 2)

    # Best hold — smaller, right-aligned inside panel
    best_str = f"BEST {best_hold:05.2f}s"
    (tw, _), _ = cv2.getTextSize(best_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, best_str,
                (px + PANEL_W - tw - 6, py + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 60), 1)

    return frame


def _process_frame(frame, pose):
    """Run pose detection on one BGR frame. Returns (annotated_frame, is_correct, pose_found)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True
    out = frame.copy()

    if not results.pose_landmarks:
        cv2.rectangle(out, (10, 10), (260, 55), (0, 0, 0), -1)
        cv2.putText(out, "No pose detected",
                    (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        return out, False, False

    landmarks  = results.pose_landmarks.landmark
    is_correct, rows = _check_frog_stand(landmarks)
    out = _annotate_frame(out, landmarks, is_correct, rows)
    return out, is_correct, True


# ── Public API ────────────────────────────────────────────────────────────────

def detect(source):
    """
    Run frog-stand detection on an image, video, or live camera.

    Parameters
    ----------
    source : str | int
        - Path to an image file  (.png / .jpg / ...)  → any key to close
        - Path to a video file   (.mp4 / .mov / ...)  → q to quit
        - Integer camera index   (0 = built-in laptop camera) → q to quit
    """
    # ── Image ──────────────────────────────────────────────────────────────
    if isinstance(source, str) and os.path.splitext(source)[1].lower() in IMAGE_EXTENSIONS:
        frame = cv2.imread(source)
        if frame is None:
            print(f"Could not load image: {source}")
            return

        with _make_pose(is_image=True) as pose:
            out, is_correct, found = _process_frame(frame, pose)

        label = "CORRECT FORM" if is_correct else ("INCORRECT FORM" if found else "No pose detected")
        print(f"[{os.path.basename(source)}] → {label}")

        cv2.imshow(f"Frog Stand — {os.path.basename(source)}", _fit_to_display(out))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ── Video or Camera ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return

    source_label = "Camera" if isinstance(source, int) else os.path.basename(source)

    hold_start = None   # time.time() when current hold began, None = resting
    best_hold  = 0.0    # longest completed or ongoing hold so far

    with _make_pose(is_image=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out, is_correct, _ = _process_frame(frame, pose)

            # ── Timer logic ───────────────────────────────────────────────
            now = time.time()
            if is_correct:
                if hold_start is None:          # just started holding
                    hold_start = now
                else:                           # still holding — update best live
                    best_hold = max(best_hold, now - hold_start)
            else:
                if hold_start is not None:      # just released — finalise best
                    best_hold  = max(best_hold, now - hold_start)
                    hold_start = None
            # ─────────────────────────────────────────────────────────────

            _draw_timer(out, hold_start, best_hold)
            cv2.imshow(f"Frog Stand — {source_label}  (q to quit)", _fit_to_display(out))

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # detect(0)                                      # laptop webcam
    detect("frog_stand/frog_stand_reference2.mp4")         # single image

    # for path in ["frog_stand/frog_complete.png", "frog_stand/Not_frog1.png", "frog_stand/Not_frog2.png"]:
    #     detect(path)

    
