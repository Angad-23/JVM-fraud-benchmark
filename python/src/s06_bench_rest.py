"""Round-trip latency against the REST service (start s04_serve.py first, in another terminal).

Label your arm so it matches how the server was started, e.g.
  python s06_bench_rest.py --label rest_flaskdev_nokeepalive --no-keepalive
  python s06_bench_rest.py --label rest_waitress_keepalive

--no-keepalive opens a new TCP connection for every request (worst case; what the old paper measured).
Default uses requests.Session() = HTTP keep-alive (what a real production client does).
NOTE: Flask's development server replies HTTP/1.0 + 'Connection: close', so keep-alive has no effect against it;
use waitress to compare keep-alive vs no keep-alive.
Use the IP address (127.0.0.1), not "localhost": on Windows "localhost" may try IPv6 first and add delay.
"""
import argparse
import time
import numpy as np
import requests
from common import load_config, sample_rows, write_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--no-keepalive", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--server-desc", default="", help="free text: how the server was started")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    b = cfg["benchmark"]
    n = a.n or b["measured_requests"]
    url = f"http://{a.host or b['service_host']}:{a.port or b['service_port']}/predict"
    rows = sample_rows(cfg, n)
    sess = None if a.no_keepalive else requests.Session()
    post = (lambda body: requests.post(url, json=body, timeout=30)) if sess is None \
        else (lambda body: sess.post(url, json=body, timeout=30))

    for i in range(b["warmup_requests"]):
        post({"features": rows[i % len(rows)].tolist()}).raise_for_status()
    lat = np.empty(n)
    for i, r in enumerate(rows):
        body = {"features": r.tolist()}
        t0 = time.perf_counter_ns()
        resp = post(body)
        resp.json()                    # include response parsing
        lat[i] = (time.perf_counter_ns() - t0) / 1000.0
        resp.raise_for_status()
    write_raw(cfg, a.label, lat, {"batch_size": 1, "warmup": b["warmup_requests"], "seed": b["seed"],
                                  "keepalive": not a.no_keepalive, "server_desc": a.server_desc, "url": url,
                                  "client": "python-requests"})


if __name__ == "__main__":
    main()
