"""The agent container's proxy: it tunnels to allowed hosts and refuses everything else.

These run the proxy on localhost in a thread, with a local echo server as the allowed
host. No Docker and no network.
"""

import socket
import threading

import pytest

from oumi_weights_lab.proving import egress


@pytest.fixture
def echo():
    """A local TCP server that echoes what it receives; yields its port."""
    server = socket.create_server(("127.0.0.1", 0))
    port = server.getsockname()[1]
    stop = threading.Event()

    def serve():
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            with connection:
                while data := connection.recv(1024):
                    connection.sendall(data)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield port
    stop.set()
    thread.join()
    server.close()


@pytest.fixture
def proxy(echo):
    """A proxy on localhost that allows only the echo server; yields (port, log lines)."""
    lines = []
    allowed = egress.parse_allow([f"127.0.0.1:{echo}", "api.anthropic.com"])
    server = egress.Proxy(("127.0.0.1", 0), allowed, log=lines.append)
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    yield server.server_address[1], lines
    server.shutdown()
    server.server_close()
    thread.join()


def ask(port, request: bytes) -> tuple[socket.socket, bytes]:
    """Send ``request`` to the proxy; return the open socket and the response head."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.sendall(request)
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = sock.recv(1024)
        if not chunk:
            break
        head += chunk
    return sock, head


def test_an_allowed_host_gets_a_tunnel(proxy, echo):
    port, lines = proxy
    sock, head = ask(port, f"CONNECT 127.0.0.1:{echo} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    with sock:
        assert head.startswith(b"HTTP/1.1 200 ")
        sock.sendall(b"hello through the tunnel")
        assert sock.recv(1024) == b"hello through the tunnel"
    assert lines == [f"opened CONNECT 127.0.0.1:{echo}"]


@pytest.mark.parametrize(
    "target",
    [
        "huggingface.co:443",
        "api.anthropic.com.example.com:443",
        "example.com:443",
        "api.anthropic.com:80",  # an allowed host on a port the list does not name
        "127.0.0.1:1",
        "user@api.anthropic.com:443",
    ],
)
def test_any_other_host_is_refused_at_once(proxy, target):
    port, lines = proxy
    sock, head = ask(port, f"CONNECT {target} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    with sock:
        while chunk := sock.recv(1024):  # the proxy closes the connection after its answer
            head += chunk
    assert head.startswith(b"HTTP/1.1 403 ") and head.endswith(b"\r\n\r\n403 Forbidden\n")
    assert lines == [f"refused CONNECT {target}"]


@pytest.mark.parametrize(
    "request_line",
    [
        "GET http://huggingface.co/openai-community/gpt2/resolve/main/config.json HTTP/1.1",
        "GET http://api.anthropic.com/ HTTP/1.1",
        "POST http://example.com/ HTTP/1.1",
    ],
)
def test_plain_http_is_refused_whatever_the_host(proxy, request_line):
    port, lines = proxy
    sock, head = ask(port, f"{request_line}\r\nHost: x\r\n\r\n".encode())
    with sock:
        assert head.startswith(b"HTTP/1.1 405 ")
    assert lines == [f"refused {' '.join(request_line.split()[:2])}"]


def test_a_malformed_request_is_refused(proxy):
    port, lines = proxy
    sock, head = ask(port, b"NONSENSE\r\n\r\n")
    with sock:
        assert head.startswith(b"HTTP/1.1 400 ")
    assert lines == []


def test_the_allow_list_parses_hosts_and_ports():
    allowed = egress.parse_allow(["API.Anthropic.com.", "example.org:8443"])
    assert allowed == {("api.anthropic.com", 443), ("example.org", 8443)}
    assert egress.permitted("api.anthropic.com:443", allowed)
    assert egress.permitted("API.ANTHROPIC.COM.:443", allowed)
    assert not egress.permitted("example.org:443", allowed)
    for bad in ["", ":443", "host:0", "host:99999", "a/b", "a b"]:
        with pytest.raises(ValueError):
            egress.parse_allow([bad])


def test_the_proxy_imports_only_the_standard_library():
    """The runner passes this file's source to ``python -c`` in an image without the lab."""
    import ast
    import sys
    from pathlib import Path

    tree = ast.parse(Path(egress.__file__).read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports"
            modules.add(node.module.split(".")[0])
    assert modules <= set(sys.stdlib_module_names) | {"__future__"}
