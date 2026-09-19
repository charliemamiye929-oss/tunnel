# mcp-tunnel

Running a local MCP server as a Claude **custom connector**, over a Cloudflare
tunnel — so it works in Claude chats on the web, desktop and phone, not only in
a terminal.

Everything here was written while getting one working. The troubleshooting
section is the part worth keeping: every entry in it is a failure that actually
happened, with the symptom it showed rather than the cause it had.

---

## What you are building

```
  Claude (cloud)                    your Mac
        |                    +---------------------------+
        |   HTTPS            |  cloudflared  -> :8787    |
        +------------------->|       tunnel     MCP srv  |
   https://mcp.you.com/mcp   +---------------------------+
```

Claude's cloud makes the connection, so the URL has to be reachable from the
public internet. `localhost`, a VPN, or anything behind a firewall cannot work,
however well it works in your browser. The tunnel exists to give a laptop a
public HTTPS address without opening a port.

Two processes must be running: your MCP server, and the tunnel. Close either
and the connector goes dead.

## What your server has to speak

**Streamable HTTP**, not stdio. A stdio server — the kind a `.mcp.json` launches
as a subprocess — cannot be a connector at all; a connector is a URL, not a
command. If your server only speaks stdio, that is the first thing to fix.

One endpoint path, handling:

| method | behaviour |
|---|---|
| `POST` | JSON-RPC message in; a JSON-RPC object back for a request, `202` with no body for a notification |
| `GET` | an SSE stream, or `405` if the server never pushes anything |
| `DELETE` | `204` — end of session |
| `OPTIONS` | `204` with CORS and `Allow` headers |
| `HEAD` | `200` |

The last two matter more than the spec suggests: see troubleshooting.

---

## Path A — a URL in five minutes (temporary)

```bash
brew install cloudflared

# terminal 1
<start your MCP server on :8787>

# terminal 2
cloudflared tunnel --url http://localhost:8787
```

It prints `https://<random-words>.trycloudflare.com`. Your connector URL is that
plus your endpoint path, e.g. `https://random-words.trycloudflare.com/mcp`.

Good for proving it works. The URL changes every single time the tunnel starts,
so you re-edit the connector on every restart.

## Path B — a URL that never changes

Needs a domain whose DNS is on a Cloudflare account (the domain can be bought
anywhere; Cloudflare's free plan is enough).

1. Cloudflare dashboard → **Domains** → **Onboard a domain** (this used to be
   called "Add a site"). Enter the apex domain, pick Free.
2. At your registrar, replace the nameservers with the two Cloudflare gives you.
3. Wait until Cloudflare shows the domain **Active**. Do not continue before
   that — step 4 fails with a confusing error.

```bash
cloudflared tunnel login                              # browser, pick the domain
cloudflared tunnel create mcp
cloudflared tunnel route dns mcp mcp.yourdomain.com
TUNNEL_HOSTNAME=mcp.yourdomain.com scripts/tunnel-up.sh
```

Connector URL: `https://mcp.yourdomain.com/mcp`, today and in a year.

## Path C — a second machine

Give every machine **its own tunnel and its own hostname**. Two machines
running the same tunnel become replicas of it, and Cloudflare splits requests
between them at random — different caches, inconsistent answers, and no error
anywhere to explain it.

If the second machine's owner should not have your Cloudflare login, create the
tunnel and DNS record yourself:

```bash
cloudflared tunnel create partner
cloudflared tunnel route dns partner mcp2.yourdomain.com
```

Send them the credentials file it writes (`~/.cloudflared/<uuid>.json`) the way
you would send a password — anyone holding it can run your tunnel. They run:

```bash
TUNNEL_HOSTNAME=mcp2.yourdomain.com TUNNEL_NAME=<uuid> \
TUNNEL_CREDENTIALS=~/.cloudflared/<uuid>.json scripts/tunnel-up.sh
```

---

## Adding it in Claude

Settings → Connectors → **Add custom connector**

| field | value |
|---|---|
| Name | anything |
| MCP server URL | `https://mcp.yourdomain.com/mcp` — the full path, not just the host |
| Authentication | **No sign-in**, unless your server really does OAuth |
| OAuth client | leave alone |
| Request headers | leave empty |

Claude checks the server as you add it. If that check fails but you know the
server is up, **Continue anyway** still configures it — the check is mostly
about detecting a sign-in flow.

---

## One command, and never running it again

```bash
scripts/tunnel install     # once: adds `tunnel` to your shell
tunnel                     # everything: setup, start, print the URL
```

On a machine with no settings, `tunnel` asks four questions, saves them to
`~/.mcp-tunnel.conf`, installs `cloudflared` if it is missing, starts the server
and tunnel **detached** — closing the terminal does not kill them — and then
checks the public URL end to end instead of assuming two live processes mean a
working connector.

```
tunnel            start, or say it is already up
tunnel stop       stop both
tunnel status     what is running, and what the public address answers
tunnel url        print the connector URL
tunnel setup      change the saved settings
tunnel logs       last lines of the server log
```

`scripts/tunnel-up.sh` and `tunnel-down.sh` are the plumbing underneath, if you
would rather wire it up yourself.

To start both at login and restart them if they die, edit the two paths in
`scripts/com.mcp-tunnel.plist`, then:

```bash
cp scripts/com.mcp-tunnel.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.mcp-tunnel.plist
```

Nothing local survives the Mac sleeping — the connector is down while the
machine is. For genuinely always-on, run the server on something that stays
awake (a small VM, Fly, Railway) and drop the tunnel entirely.

---

## Troubleshooting

Every row here is something that actually happened.

| symptom | cause | fix |
|---|---|---|
| Connector check: **"Not found", 400** | the server answered 400 to Claude's probe. Three separate causes, all real: `OPTIONS`/`HEAD` falling through to Python's `http.server` default **501**; an empty-body probe POST hitting a JSON parser and returning **400**; and a `MCP-Protocol-Version` header newer than the server's list being rejected with **400** | answer `OPTIONS` with 204 and `HEAD` with 200; treat an empty POST body as a reachability probe, not a message; accept any protocol version and negotiate in `initialize` |
| Connects, then no tool ever works | the request body arrived **chunked** and the server only read `Content-Length`, so the body was empty and the message never ran — while still answering 200 | read chunked bodies |
| Every request after the first fails to parse | a reply was sent without draining the request body; the unread bytes sit at the head of the next request on a keep-alive connection | always read the body before replying, including on 401/404 paths |
| Some concurrent calls fail with `SQLite objects created in a thread can only be used in that same thread` | `http.server` gives each request its own thread; the server's state was written for one | `check_same_thread=False` plus a lock, or a connection per thread. Sequential requests hide this completely |
| **502** from the tunnel URL | the tunnel is up, your local server is not | restart the server; `502` always means Cloudflare reached the tunnel and the tunnel could not reach the origin |
| **1033** or **530** | the tunnel itself is down | restart `cloudflared` |
| `zsh: command not found` | the tool isn't installed, or the package isn't installed as a command | `brew install cloudflared`; run a Python package with `python3 -m <package>` from inside its directory |
| `pip install -r requirements.txt` fails and nothing installs | one unsatisfiable pin aborts the whole transaction — an optional dependency with no version for your Python takes the required ones down with it | split optional dependencies into a second requirements file |
| A typed command appears in the log of a running server | it went to the server's stdin, not the shell — the server was still running | Ctrl-C first |
| Connector works from one machine, flaky from another | two machines running the same tunnel name | one tunnel and one hostname per machine |

Two commands worth knowing:

```bash
curl https://mcp.yourdomain.com/health     # is the whole path alive
tail -20 ~/.mcp-tunnel/server.log          # why the server didn't start
```

---

## Security

A tunnel URL is the public internet. Before pointing it at anything:

- **Anyone who has the URL can call every tool.** A random `trycloudflare.com`
  name is obscure, not private; a domain you chose is guessable.
- Require a token (`Authorization: Bearer …`) if your connector can send
  headers. If it cannot, put a random string in the path and treat the URL
  itself as the credential.
- **Refuse a missing token with 403, not 401.** In MCP a 401 means
  "authenticate with me over OAuth", so a client that gets one starts hunting
  for an authorization server — `/.well-known/oauth-protected-resource`,
  `/register` — finds nothing, and reports your connector as needing a sign-in
  that does not exist. On a server that is running perfectly.
- Serve only what you mean to. If one process can serve several MCP servers,
  publishing "all of them" may expose more than the one you wanted.
- Validate the `Origin` header and bind to `127.0.0.1`, so a web page your own
  browser opens cannot reach the server directly.
- The credentials file `cloudflared` writes is a credential. Not in a repo, not
  in a group chat.
