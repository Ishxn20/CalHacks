import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import cv2

from scrub_vision import markers
from scrub_vision.grasp import Finder
from scrub_vision.serve import Service, make_handler
from test_grasp import scene


class Fixed:
    def __init__(self, img):
        self.img = img

    def frame(self):
        return self.img.copy()


def test_detect_reports_stable_target_and_markers(tmp_path):
    img = scene(25)
    sheet = tmp_path / "m.png"
    markers.sheet(str(sheet), n=1)
    m = cv2.resize(cv2.imread(str(sheet)), (100, 100))
    img[10:110, 10:110] = m                                          # a tray marker in the corner
    out = Service(Fixed(img), Finder()).detect(None, frames=3)
    assert out["found"] and out["stable"] and out["identity"] == "unchecked"
    assert abs(out["grip_deg"] - (25 + 90 - 180)) < 2                # 115 folded into [-90, 90)
    assert "0" in out["markers"]


def test_nothing_in_view_is_not_found():
    out = Service(Fixed(scene(0, bg=90, fg=90)), Finder()).detect(["x"], frames=2)
    assert out["found"] is False


def test_http_round_trip():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Service(Fixed(scene(10)), Finder())))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/detect", data=json.dumps({"frames": 2}).encode())
        body = json.loads(urllib.request.urlopen(req, timeout=5).read())
        assert body["found"] and body["width_px"] > 20
        assert json.loads(urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_port}/markers").read())["markers"] == {}
    finally:
        httpd.shutdown()
