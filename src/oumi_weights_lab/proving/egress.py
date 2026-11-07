"""The agent container's only way out: an HTTPS proxy that tunnels to allowed hosts only.

``proving-ground run`` puts the agent's container on an internal Docker network with no
route out, and starts this proxy beside it in a second container that is on both that
network and the default bridge. The agent reaches the proxy through ``HTTPS_PROXY`` and
``HTTP_PROXY``. The proxy answers ``CONNECT host:port`` only: it opens the tunnel when the
host is on the allow list (port 443 unless the entry names another port) and refuses
everything else at once, so a blocked request fails fast instead of hanging. Plain HTTP
requests, to any host, are refused. The proxy never reads the bytes inside a tunnel.

This file uses the Python standard library only and imports nothing from the lab, so the
runner can pass its source to ``python -c`` in an image that does not hold the lab. Run it
as ``python egress.py --allow api.anthropic.com``.
"""

from __future__ import annotations

import argparse
import select
import socket
import socketserver
import sys

PORT = 3128
HEAD_LIMIT = 16 * 1024  # bytes of request line and headers the proxy reads
HEAD_TIMEOUT = 10.0  # seconds a client has to send its request
CONNECT_TIMEOUT = 10.0  # seconds to reach an allowed host
IDLE_TIMEOUT = 900.0  # seconds a tunnel may stay silent before it is closed
DEFAULT_PORT = 443


def parse_allow(entries) -> frozenset[tuple[str, int]]:
    """The allow list as ``(host, port)`` pairs. An entry is ``host`` (port 443) or
    ``host:port``. Hosts compare in lower case without a trailing dot. Refuses an empty
    host, a port outside 1 to 65535, and anything with characters a host name cannot hold."""
    allowed = set()
    for entry in entries:
        host, port = _split(entry)
        if host is None:
            raise ValueError(f"{entry!r} is not a host or host:port")
        allowed.add((host, port))
    return frozenset(allowed)


def _split(target: str) -> tuple[str | None, int]:
    host, sep, port_text = target.strip().rpartition(":")
    if not sep:
        host, port_text = port_text, str(DEFAULT_PORT)
    host = host.lower().rstrip(".")
    if not host or not port_text.isdigit() or not 0 < int(port_text) < 65536:
        return None, 0
    if not all(ch.isalnum() or ch in "-." for ch in host):
        return None, 0
    return host, int(port_text)


def permitted(target: str, allowed: frozenset[tuple[str, int]]) -> bool:
    """Whether a CONNECT target ``host:port`` is on the allow list (exact host and port)."""
    host, port = _split(target)
    return host is not None and (host, port) in allowed


def _read_head(connection: socket.socket) -> bytes | None:
    """The request line and headers, up to the blank line; None when too long or cut off."""
    connection.settimeout(HEAD_TIMEOUT)
    head = b""
    while b"\r\n\r\n" not in head:
        if len(head) > HEAD_LIMIT:
            return None
        chunk = connection.recv(4096)
        if not chunk:
            return None
        head += chunk
    return head


def _reply(connection: socket.socket, status: str) -> None:
    body = f"{status}\n".encode()
    connection.sendall(
        f"HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n".encode()
        + body
    )


def _relay(client: socket.socket, upstream: socket.socket) -> None:
    """Copy bytes both ways until one side closes or the tunnel is idle too long."""
    sockets = [client, upstream]
    for sock in sockets:
        sock.settimeout(None)
    while True:
        ready, _, broken = select.select(sockets, [], sockets, IDLE_TIMEOUT)
        if broken or not ready:
            return
        for sock in ready:
            data = sock.recv(65536)
            if not data:
                return
            (upstream if sock is client else client).sendall(data)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        client = self.request
        try:
            head = _read_head(client)
        except OSError:
            return
        if head is None:
            _reply(client, "400 Bad Request")
            return
        parts = head.split(b"\r\n", 1)[0].decode("latin-1").split()
        if len(parts) != 3 or not parts[2].startswith("HTTP/"):
            _reply(client, "400 Bad Request")
            return
        method, target = parts[0], parts[1]
        if method != "CONNECT":
            _reply(client, "405 Method Not Allowed")
            self.server.log(f"refused {method} {target}")
            return
        if not permitted(target, self.server.allowed):
            _reply(client, "403 Forbidden")
            self.server.log(f"refused CONNECT {target}")
            return
        host, port = _split(target)
        try:
            upstream = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT)
        except OSError:
            _reply(client, "502 Bad Gateway")
            self.server.log(f"failed CONNECT {target}")
            return
        with upstream:
            client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.server.log(f"opened CONNECT {target}")
            try:
                _relay(client, upstream)
            except OSError:
                pass


class Proxy(socketserver.ThreadingTCPServer):
    """The proxy server. ``allowed`` holds ``(host, port)`` pairs; ``log`` gets one line per
    request it refuses or opens (stderr by default)."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], allowed, log=None):
        self.allowed = allowed
        self.log = log or (lambda line: print(line, file=sys.stderr, flush=True))
        super().__init__(address, _Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="An HTTPS proxy that tunnels to allowed hosts.")
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--allow", action="append", required=True, help="host or host:port")
    args = parser.parse_args(argv)
    allowed = parse_allow(args.allow)
    with Proxy((args.listen, args.port), allowed) as server:
        names = ", ".join(f"{host}:{port}" for host, port in sorted(allowed))
        print(f"listening on {args.listen}:{args.port}; allowed: {names}", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
