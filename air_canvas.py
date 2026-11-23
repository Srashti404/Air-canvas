"""
Air Canvas - draw in the air with your index finger.

Install:   pip install opencv-python mediapipe numpy
Run:       python air_canvas.py

Gestures
  Index finger up only            -> DRAW
  Index + middle fingers up       -> SELECT (hover over a button at the top)
  Any other pose (e.g. fist)      -> pen lifted

Keys
  c = clear   s = save PNG   + / - = brush size   q or ESC = quit
"""

import time
import cv2
import numpy as np
import mediapipe as mp

# ---------- settings ----------
CAM_INDEX = 0
WIDTH, HEIGHT = 1280, 720
BAR_H = 90                      # height of the toolbar
SMOOTHING = 0.5                 # 0 = raw, closer to 1 = smoother but laggier
COLORS = {                      # BGR
    "Blue":   (255, 80, 40),
    "Green":  (60, 200, 60),
    "Red":    (50, 50, 240),
    "Yellow": (0, 220, 255),
    "White":  (255, 255, 255),
}
# ------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def build_buttons():
    """Toolbar buttons: one per color, then Eraser and Clear."""
    names = list(COLORS.keys()) + ["Eraser", "Clear"]
    w = WIDTH // len(names)
    return [(n, i * w, (i + 1) * w) for i, n in enumerate(names)]


def draw_toolbar(frame, buttons, active):
    for name, x1, x2 in buttons:
        if name in COLORS:
            color = COLORS[name]
            text_col = (0, 0, 0) if name in ("Yellow", "White") else (255, 255, 255)
        else:
            color = (70, 70, 70)
            text_col = (255, 255, 255)
        cv2.rectangle(frame, (x1 + 4, 8), (x2 - 4, BAR_H - 8), color, -1)
        if name == active:
            cv2.rectangle(frame, (x1 + 4, 8), (x2 - 4, BAR_H - 8), (255, 255, 255), 4)
        size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        tx = x1 + ((x2 - x1) - size[0]) // 2
        cv2.putText(frame, name, (tx, BAR_H // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_col, 2, cv2.LINE_AA)


def fingers_up(lm, handedness_label):
    """Return [thumb, index, middle, ring, pinky] as booleans."""
    up = []
    # thumb: compare x of tip vs joint, depends on which hand (image is mirrored)
    if handedness_label == "Right":
        up.append(lm[4].x > lm[3].x)
    else:
        up.append(lm[4].x < lm[3].x)
    for tip in (8, 12, 16, 20):
        up.append(lm[tip].y < lm[tip - 2].y)
    return up


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try changing CAM_INDEX.")

    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    buttons = build_buttons()
    active = "Blue"
    brush = 8
    prev = None          # previous (smoothed) fingertip position
    smooth = None
    last_select = 0.0

    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        mode = "No hand"
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            label = result.multi_handedness[0].classification[0].label
            lm = hand.landmark
            up = fingers_up(lm, label)
            _, index, middle, ring, pinky = up

            raw = np.array([lm[8].x * WIDTH, lm[8].y * HEIGHT])
            smooth = raw if smooth is None else SMOOTHING * smooth + (1 - SMOOTHING) * raw
            x, y = int(smooth[0]), int(smooth[1])

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            if index and middle and not ring and not pinky:
                # ---- SELECT mode ----
                mode = "Select"
                prev = None
                cv2.circle(frame, (x, y), 18, (255, 255, 255), 2)
                if y < BAR_H and time.time() - last_select > 0.4:
                    for name, x1, x2 in buttons:
                        if x1 <= x < x2:
                            last_select = time.time()
                            if name == "Clear":
                                canvas[:] = 0
                            else:
                                active = name
                            break

            elif index and not middle and not ring and not pinky:
                # ---- DRAW mode ----
                mode = "Drawing" if active != "Eraser" else "Erasing"
                if y > BAR_H:  # don't draw on the toolbar
                    if active == "Eraser":
                        cv2.circle(canvas, (x, y), brush * 3, (0, 0, 0), -1)
                        cv2.circle(frame, (x, y), brush * 3, (200, 200, 200), 2)
                        prev = None
                    else:
                        color = COLORS[active]
                        if prev is not None:
                            cv2.line(canvas, prev, (x, y), color, brush, cv2.LINE_AA)
                        prev = (x, y)
                        cv2.circle(frame, (x, y), brush // 2 + 4, color, -1)
                else:
                    prev = None
            else:
                mode = "Pen up"
                prev = None
        else:
            prev = None
            smooth = None

        # merge canvas onto camera frame
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        out = cv2.add(bg, fg)

        draw_toolbar(out, buttons, active)
        cv2.putText(out, f"Mode: {mode}   Brush: {brush}", (15, HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Air Canvas", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            canvas[:] = 0
        elif key == ord("s"):
            # save drawing on a white background
            white = np.full_like(canvas, 255)
            drawing = np.where(mask[..., None] > 0, canvas, white)
            fname = f"air_canvas_{int(time.time())}.png"
            cv2.imwrite(fname, drawing)
            print("Saved", fname)
        elif key in (ord("+"), ord("=")):
            brush = min(40, brush + 2)
        elif key in (ord("-"), ord("_")):
            brush = max(2, brush - 2)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
