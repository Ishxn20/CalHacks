"""Save the wrist camera's view to hw_check/live.jpg once a second, with the 9x6 checkerboard corners drawn
when found, so someone without camera access can look. Run from Terminal.app; Ctrl-C to stop."""
import sys
import time
from pathlib import Path

import cv2

out = Path(__file__).with_name("live.jpg")
cap = cv2.VideoCapture(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
while True:
    ok, f = cap.read()
    if not ok:
        time.sleep(0.1)
        continue
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(g, (9, 6), flags=cv2.CALIB_CB_FAST_CHECK)
    if found:
        cv2.drawChessboardCorners(f, (9, 6), corners, found)
    cv2.putText(f, f"board {'FOUND' if found else 'not found'}  {time.strftime('%H:%M:%S')}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0) if found else (0, 0, 255), 3)
    cv2.imwrite(str(out.with_suffix(".tmp.jpg")), f)
    out.with_suffix(".tmp.jpg").replace(out)
    print(f"\rboard {'FOUND    ' if found else 'not found'}", end="", flush=True)
    time.sleep(1.0)
