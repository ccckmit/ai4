"""
Tests for world.render.server module.
"""

import json
import socket
import struct
import time

import pytest

from world.render import server as rs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def server():
    """Start server on random port, yield (port, stop_fn)."""
    # find random available port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    rs.start_server(port)
    time.sleep(0.2)
    yield port
    rs.stop_server()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _http_get(port: int, path: str = "/") -> bytes:
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode())
    s.settimeout(3)
    data = s.recv(65536)
    s.close()
    return data


def _ws_connect(port: int) -> socket.socket:
    """Open a WebSocket connection, return connected socket."""
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    key = "dGhlIHNhbXBsZSBub25jZQ=="
    req = (
        "GET / HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    )
    s.sendall(req.encode())
    s.settimeout(3)
    resp = s.recv(4096)
    assert b"101 Switching Protocols" in resp, f"WS handshake failed: {resp[:200]}"
    return s


def _read_ws_frame(s: socket.socket) -> bytes:
    """Read one WebSocket text frame payload."""
    b0 = s.recv(1)
    assert b0[0] & 0x80, "not a final frame"
    opcode = b0[0] & 0x0F
    assert opcode == 1, f"expected text frame (1), got {opcode}"
    b1 = s.recv(1)
    length = b1[0] & 0x7F
    if length == 126:
        length = struct.unpack(">H", s.recv(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", s.recv(8))[0]
    return s.recv(length)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRenderServer:

    def test_http_serves_html(self, server: int):
        port = server
        data = _http_get(port)
        assert b"HTTP/1.1 200 OK" in data, f"bad response: {data[:100]}"
        assert b"<!DOCTYPE html>" in data, "should serve viewer.html"
        assert b"CartPole" in data, "HTML should contain CartPole"

    def test_ws_handshake(self, server: int):
        port = server
        s = _ws_connect(port)
        s.close()

    def test_send_frame_received(self, server: int):
        port = server
        ws = _ws_connect(port)
        time.sleep(0.1)

        rs.send_frame(x=0.5, theta=0.1, steps=42, reward=1.0, done=False)
        time.sleep(0.1)

        payload = _read_ws_frame(ws)
        obj = json.loads(payload.decode())
        assert obj["x"] == 0.5
        assert obj["theta"] == 0.1
        assert obj["steps"] == 42
        assert obj["reward"] == 1.0
        assert obj["done"] is False
        ws.close()

    def test_multiple_frames(self, server: int):
        port = server
        ws = _ws_connect(port)
        time.sleep(0.1)

        for step in range(5):
            rs.send_frame(x=step * 0.1, theta=0.0, steps=step, reward=1.0, done=False)
            time.sleep(0.05)

        # drain all buffered frames, keep the last one
        last_payload = b""
        ws.settimeout(0.5)
        while True:
            try:
                last_payload = _read_ws_frame(ws)
            except (socket.timeout, OSError, AssertionError):
                break
        ws.settimeout(3)
        assert last_payload, "expected at least one frame"
        obj = json.loads(last_payload.decode())
        assert obj["steps"] >= 4
        ws.close()

    def test_done_flag_in_frame(self, server: int):
        port = server
        ws = _ws_connect(port)
        time.sleep(0.1)

        rs.send_frame(x=0.0, theta=0.0, steps=1, reward=1.0, done=True)
        time.sleep(0.1)

        payload = _read_ws_frame(ws)
        obj = json.loads(payload.decode())
        assert obj["done"] is True
        ws.close()

    def test_http_404(self, server: int):
        port = server
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        s.sendall(b"GET /nonexistent HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        s.settimeout(3)
        data = s.recv(65536)
        s.close()
        assert b"200 OK" in data, "any path gets the HTML page"

    def test_multiple_ws_clients(self, server: int):
        """Multiple WS clients all receive frames."""
        port = server
        n = 3
        clients = [_ws_connect(port) for _ in range(n)]
        time.sleep(0.2)

        rs.send_frame(x=1.0, theta=0.2, steps=10, reward=0.5, done=False)
        time.sleep(0.15)

        for i, c in enumerate(clients):
            payload = _read_ws_frame(c)
            obj = json.loads(payload.decode())
            assert obj["x"] == 1.0, f"client {i} got wrong x"
            c.close()
