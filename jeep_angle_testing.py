import cv2, numpy as np, math, argparse, sys
from pathlib import Path

def angle_from_red_contour(image_bgr: np.ndarray):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Red mask (tweak if your lighting/theme changes)
    lower_red1 = np.array([0, 90, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 90, 60])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 200:
        return None

    (cx, cy), (w, h), theta = cv2.minAreaRect(cnt)
    angle = theta + 90 if w < h else theta
    if angle < -90: angle += 180
    if angle > 90:  angle -= 180
    return (float(angle), (int(cx), int(cy)), cnt)

def annotate_with_direction_line(image_bgr: np.ndarray):
    res = angle_from_red_contour(image_bgr)
    annotated = image_bgr.copy()
    if res is None:
        cv2.putText(annotated, "Red jeep not found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
        return annotated, None

    angle, (cx, cy), cnt = res
    h, w = image_bgr.shape[:2]
    cv2.drawContours(annotated, [cnt], -1, (0,255,0), 2)

    rad = math.radians(angle)
    length = int(max(w, h) * 1.2)
    dx, dy = int(math.cos(rad)*length), int(math.sin(rad)*length)
    cv2.line(annotated, (cx-dx, cy-dy), (cx+dx, cy+dy), (0,0,255), 3)
    cv2.circle(annotated, (cx, cy), 6, (255,255,255), -1)

    label = f"Angle: {angle:.1f} deg"
    cv2.putText(annotated, label, (cx+10, max(30, cy-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(annotated, label, (cx+10, max(30, cy-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return annotated, angle

def main():
    ap = argparse.ArgumentParser(description="Detect jeep angle and draw direction line")
    ap.add_argument("input", help="Path to input image (or 0 for webcam)")
    ap.add_argument("-o", "--output", help="Save annotated image/video to this path")
    args = ap.parse_args()

    if args.input == "0":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("Could not open webcam.")
        while True:
            ok, frame = cap.read()
            if not ok: break
            annotated, angle = annotate_with_direction_line(frame)
            cv2.imshow("Jeep Angle", annotated)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        img = cv2.imread(args.input)
        if img is None:
            sys.exit(f"Could not read {args.input}")
        annotated, angle = annotate_with_direction_line(img)
        print(f"Angle (deg): {angle}")
        if args.output:
            cv2.imwrite(args.output, annotated)
        cv2.imshow("Jeep Angle", annotated)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
