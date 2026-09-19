"""Protocol-level tests for the MCP server.

Each test here corresponds to a failure that actually broke a connector:
wrong status for OPTIONS/HEAD, a 400 on an empty probe, a rejected protocol
version, a chunked body read as empty, and sqlite used across threads.
"""
import json
import threading
import urllib.error
import urllib.request
from http.client import HTTPConnection

import pytest

from evotrader import mcp_server
from evotrader.mcp_server import MCPHandler, handle_message


@pytest.fixture(scope="module")
def server():
    from http.server import ThreadingHTTPServer

    MCPHandler.endpoint = "/mcp"
    MCPHandler.token = ""
    MCPHandler.allowed_origins = ()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), MCPHandler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def request(addr, method, path="/mcp", body=None, headers=None):
    conn = HTTPConnection(addr, timeout=10)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    payload = resp.read()
    conn.close()
    return resp.status, payload


def test_options_is_204_not_501(server):
    status, _ = request(server, "OPTIONS")
    assert status == 204


def test_head_is_200_not_501(server):
    status, _ = request(server, "HEAD")
    assert status == 200


def test_empty_post_probe_is_not_400(server):
    status, body = request(server, "POST", body=b"")
    assert status == 200
    assert json.loads(body)["status"] == "ok"


def test_get_is_405_no_stream(server):
    status, _ = request(server, "GET")
    assert status == 405


def test_delete_is_204(server):
    status, _ = request(server, "DELETE")
    assert status == 204


def test_health(server):
    status, body = request(server, "GET", "/health")
    assert status == 200
    assert json.loads(body)["status"] == "ok"


def test_unknown_protocol_version_is_negotiated_not_rejected(server):
    msg = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": "2099-01-01"}}).encode()
    status, body = request(server, "POST", body=msg,
                           headers={"Content-Type": "application/json",
                                    "MCP-Protocol-Version": "2099-01-01"})
    assert status == 200
    assert json.loads(body)["result"]["protocolVersion"] == mcp_server.DEFAULT_PROTOCOL


def test_chunked_body_is_read(server):
    """A chunked request must actually run, not silently return an empty result."""
    payload = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/list"}).encode()
    conn = HTTPConnection(server, timeout=10)
    conn.putrequest("POST", "/mcp")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Transfer-Encoding", "chunked")
    conn.endheaders()
    conn.send(b"%x\r\n%s\r\n" % (len(payload), payload))
    conn.send(b"0\r\n\r\n")
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    assert resp.status == 200
    assert data["id"] == 7
    assert len(data["result"]["tools"]) == len(mcp_server.TOOLS)


def test_body_is_drained_so_keepalive_survives(server):
    """A reply that skips the body leaves bytes at the head of the next request."""
    conn = HTTPConnection(server, timeout=10)
    for i in (1, 2, 3):
        msg = json.dumps({"jsonrpc": "2.0", "id": i, "method": "ping"}).encode()
        # A 404 path still has to read the body before answering.
        path = "/nope" if i == 2 else "/mcp"
        conn.request("POST", path, body=msg,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        assert data is not None
    conn.close()


def test_notification_gets_202_and_no_body(server):
    msg = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode()
    status, body = request(server, "POST", body=msg,
                           headers={"Content-Type": "application/json"})
    assert status == 202
    assert body == b""


def test_concurrent_calls_do_not_trip_sqlite_threading(server):
    import concurrent.futures as cf

    def call(i):
        msg = json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                          "params": {"name": "list_runs", "arguments": {"limit": 5}}}).encode()
        req = urllib.request.Request(f"http://{server}/mcp", data=msg,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(call, range(40)))
    for data in results:
        assert "error" not in data
        assert data["result"]["isError"] is False


def test_bearer_token_is_enforced(server):
    MCPHandler.token = "secret"
    try:
        msg = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()
        status, _ = request(server, "POST", body=msg,
                            headers={"Content-Type": "application/json"})
        assert status == 401
        status, _ = request(server, "POST", body=msg,
                            headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer secret"})
        assert status == 200
    finally:
        MCPHandler.token = ""


def test_unknown_method_and_tool_are_clean_errors():
    assert handle_message({"jsonrpc": "2.0", "id": 1,
                           "method": "nope/nope"})["error"]["code"] == -32601
    out = handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": "nope", "arguments": {}}})
    assert out["error"]["code"] == -32602


def test_every_tool_is_listed_with_a_schema():
    for tool in mcp_server._public_tools():
        assert tool["inputSchema"]["type"] == "object"
        assert tool["description"]
        assert "handler" not in tool
