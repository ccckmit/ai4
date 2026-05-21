"""
WebSocket + HTTP server for AI4 render viewer.

Single-port design: HTTP page served on GET, WebSocket upgrade
on the same port for real-time frame push. No third-party deps.
"""

import base64
import hashlib
import json
import os
import socket
import struct
import threading
import time
import webbrowser

_HERE = os.path.dirname(os.path.abspath(__file__))
_HTML_PATH = os.path.join(_HERE, "viewer.html")
_WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_http_server: socket.socket | None = None
_threads: list[threading.Thread] = []
_running = threading.Event()
_latest_frame: str = ""


def _load_html() -> bytes:
    with open(_HTML_PATH, "rb") as f:
        return f.read()


def _compute_accept(key: str) -> str:
    return base64.b64encode(
        hashlib.sha1((key + _WS_MAGIC).encode()).digest()
    ).decode()


def _send_ws_frame(conn: socket.socket, data: bytes) -> None:
    frame = bytearray()
    frame.append(0x81)
    n = len(data)
    if n < 126:
        frame.append(n)
    elif n < 65536:
        frame.extend(b"\x7e" + struct.pack(">H", n))
    else:
        frame.extend(b"\x7f" + struct.pack(">Q", n))
    frame.extend(data)
    try:
        conn.sendall(bytes(frame))
    except OSError:
        pass


def _try_recv(conn: socket.socket, bufsize: int = 8192) -> bytes | None:
    """Read from socket with timeout. Returns None on timeout / error."""
    try:
        return conn.recv(bufsize)
    except socket.timeout:
        return None
    except OSError:
        return None


def _serve_html(conn: socket.socket, html: bytes) -> None:
    resp = (
        "HTTP/1.1 200 OK\r\n"
        f"Content-Length: {len(html)}\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Connection: close\r\n\r\n"
    ).encode() + html
    try:
        conn.sendall(resp)
    except OSError:
        pass
    finally:
        conn.close()


def _do_ws_handshake(conn: socket.socket, data: bytes) -> bool:
    """Read WS upgrade request from `data`, send 101 response.
    Returns True if handshake succeeded."""
    req = data.decode("utf-8", errors="replace")
    key = ""
    for line in req.split("\r\n"):
        if line.lower().startswith("sec-websocket-key:"):
            key = line.split(":", 1)[1].strip()
            break
    if not key:
        conn.close()
        return False
    accept = _compute_accept(key)
    resp = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    try:
        conn.sendall(resp.encode())
        return True
    except OSError:
        conn.close()
        return False


def _push_frames(conn: socket.socket) -> None:
    global _latest_frame
    last_sent = ""
    conn.settimeout(0.001)  # non-blocking-ish for ping/close drain
    while _running.is_set():
        f = _latest_frame
        if f and f != last_sent:
            _send_ws_frame(conn, f.encode())
            last_sent = f
        # drain any inbound data (pings / closes) without blocking
        _try_recv(conn, 4096)
        time.sleep(0.033)
    conn.close()


def _serve(html: bytes) -> None:
    while _running.is_set():
        try:
            conn, addr = _http_server.accept()
            conn.settimeout(5.0)
        except socket.timeout:
            continue
        except OSError:
            break

        data = _try_recv(conn)
        if data is None:
            conn.close()
            continue

        if b"upgrade: websocket" in data.lower():
            if _do_ws_handshake(conn, data):
                t = threading.Thread(target=_push_frames, args=(conn,), daemon=True)
                t.start()
                _threads.append(t)
        else:
            t = threading.Thread(target=_serve_html, args=(conn, html), daemon=True)
            t.start()


def start_server(port: int = 8080) -> None:
    global _http_server
    if _http_server is not None:
        return

    _running.set()
    html = _load_html()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(5)
    sock.settimeout(1.0)
    _http_server = sock

    print(f"[render] http://localhost:{port}")

    def _open_browser():
        time.sleep(0.3)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=_open_browser, daemon=True).start()

    t = threading.Thread(target=_serve, args=(html,), daemon=True)
    t.start()
    _threads.append(t)


def send_frame(x: float, theta: float, steps: int, reward: float, done: bool) -> None:
    global _latest_frame, _http_server
    if _http_server is None:
        start_server()
    _latest_frame = json.dumps({
        "x": round(x, 4),
        "theta": round(theta, 4),
        "steps": steps,
        "reward": round(reward, 2),
        "done": done,
    })


def stop_server() -> None:
    global _http_server, _latest_frame
    _running.clear()
    if _http_server is not None:
        try:
            _http_server.close()
        except OSError:
            pass
        _http_server = None
    _latest_frame = ""
    for t in _threads:
        if t.is_alive():
            t.join(timeout=1)
    _threads.clear()
