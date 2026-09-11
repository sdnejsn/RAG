"""
极简 RESP（Redis 协议）测试服务器 —— 仅用于验证 cache.py 的 Redis 代码路径。

为什么需要它？
  要真实验证 _RedisBackend，需要（a）python `redis` 客户端包，（b）一个 Redis 服务。
  本机有 redis-server，但客户端包未安装且当前沙箱不允许写入 Python 环境。
  这个脚本用标准库 socket 直接讲 RESP 协议，配合 `redis` 客户端的 RESP 解析器，
  就能把 _RedisBackend 的每条代码路径真实跑一遍，并把它发出的命令记录下来，
  用于断言"实现是否符合预期"（例如必须用 SCAN 而不是会阻塞实例的 KEYS）。

它**不是** Redis 的替代品，只实现了本测试用到的命令：
  PING GET SET(EX) INCR DELETE SCAN(COUNT/MATCH) EXISTS KEYS FLUSHALL
  以及 redis-py 连接时的 CLIENT SETINFO / CLIENT SETNAME（返回 OK 即可）

用法：
    python tools/mini_redis_resp.py 6399            # 启动在 6399 端口
    （另开一个终端）CACHE_BACKEND=redis REDIS_URL=redis://127.0.0.1:6399/0 python verify_cache.py
"""
import fnmatch
import socket
import sys
import threading
import time
from typing import Dict, List

HOST = "127.0.0.1"
DEFAULT_PORT = 6399

_data: Dict[str, str] = {}          # 简化：只存字符串（版本号、JSON）
_expire: Dict[str, float] = {}
_lock = threading.Lock()
_commands: List[str] = []           # 记录收到的命令，便于事后断言


def _encode(value: str) -> bytes:
    return value.encode("utf-8")


def _simple(text: str) -> bytes:
    return b"+" + _encode(text) + b"\r\n"


def _error(text: str) -> bytes:
    return b"-" + _encode(text) + b"\r\n"


def _integer(n: int) -> bytes:
    return b":" + str(n).encode() + b"\r\n"


def _bulk(value) -> bytes:
    if value is None:
        return b"$-1\r\n"
    data = _encode(value) if isinstance(value, str) else value
    return b"$" + str(len(data)).encode() + b"\r\n" + data + b"\r\n"


def _array(items: List[bytes]) -> bytes:
    out = b"*" + str(len(items)).encode() + b"\r\n"
    for item in items:
        out += _bulk(item)
    return out


def _is_expired(key: str) -> bool:
    exp = _expire.get(key)
    if exp is not None and exp < time.time():
        _data.pop(key, None)
        _expire.pop(key, None)
        return True
    return False


def _to_str(value: bytes) -> str:
    return value.decode("utf-8", "replace")


def dispatch(args: List[bytes]) -> bytes:
    """执行一条命令。返回 RESP 响应字节。"""
    if not args:
        return _error("empty command")
    cmd = _to_str(args[0]).upper()
    argv = [_to_str(a) for a in args[1:]]
    _commands.append(" ".join([cmd] + argv))

    with _lock:
        if cmd == "PING":
            return _simple("PONG")

        if cmd in ("CLIENT", "SELECT", "AUTH"):
            return _simple("OK")

        if cmd == "GET":
            key = argv[0]
            _is_expired(key)
            return _bulk(_data.get(key))

        if cmd == "SET":
            key, value = argv[0], argv[1]
            ttl = None
            for i, opt in enumerate(argv[2:], start=2):
                if opt.upper() in ("EX", "PX") and i + 1 < len(argv):
                    n = float(argv[i + 1])
                    ttl = n if opt.upper() == "EX" else n / 1000.0
            _data[key] = value
            if ttl is not None:
                _expire[key] = time.time() + ttl
            else:
                _expire.pop(key, None)
            return _simple("OK")

        if cmd == "INCR":
            key = argv[0]
            _is_expired(key)
            current = int(_data.get(key, "0")) + 1
            _data[key] = str(current)
            _expire.pop(key, None)
            return _integer(current)

        if cmd == "DELETE":
            n = 0
            for key in argv:
                if _data.pop(key, None) is not None:
                    _expire.pop(key, None)
                    n += 1
            return _integer(n)

        if cmd == "EXISTS":
            return _integer(sum(1 for k in argv if not _is_expired(k) and k in _data))

        if cmd == "KEYS":
            pattern = argv[0] if argv else "*"
            keys = [k for k in list(_data) if not _is_expired(k) and fnmatch.fnmatchcase(k, pattern)]
            # 故意记录：客户端如果用了 KEYS，_commands 里会出现这条，测试会断言它不存在
            return _array([_encode(k) for k in keys])

        if cmd == "SCAN":
            # 简化实现：一次性返回全部匹配 key，cursor 返回 0 表示遍历结束
            pattern, count = "*", 10
            for i, opt in enumerate(argv):
                if opt.upper() == "MATCH" and i + 1 < len(argv):
                    pattern = argv[i + 1]
                if opt.upper() == "COUNT" and i + 1 < len(argv):
                    count = int(argv[i + 1])
            keys = [k for k in list(_data) if not _is_expired(k) and fnmatch.fnmatchcase(k, pattern)]
            batch = keys[:count]
            return b"*2\r\n" + _bulk("0") + _array([_encode(k) for k in batch])

        if cmd == "FLUSHALL":
            _data.clear()
            _expire.clear()
            return _simple("OK")

        if cmd == "DBSIZE":
            return _integer(len(_data))

    return _error(f"unknown command {cmd}")


def read_command(conn) -> List[bytes]:
    """读一条命令，支持 RESP 数组（redis-py 用这个）。"""
    line = b""
    while not line.endswith(b"\r\n"):
        chunk = conn.recv(1)
        if not chunk:
            return []
        line += chunk
    if not line.startswith(b"*"):
        return [line.strip()]          # inline 命令
    n = int(line[1:].strip())
    args = []
    for _ in range(n):
        header = b""
        while not header.endswith(b"\r\n"):
            chunk = conn.recv(1)
            if not chunk:
                return []
            header += chunk
        length = int(header[1:].strip())
        payload = b""
        while len(payload) < length + 2:
            payload += conn.recv(length + 2 - len(payload))
        args.append(payload[:length])
    return args


def handle(conn) -> None:
    try:
        while True:
            args = read_command(conn)
            if not args:
                return
            conn.sendall(dispatch(args))
    except OSError:
        pass
    finally:
        conn.close()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(16)
    print(f"[mini-redis] listening on {HOST}:{port}", flush=True)
    try:
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        print("[mini-redis] stopped", flush=True)
        with open("mini_redis_commands.log", "w", encoding="utf-8") as f:
            f.write("\n".join(_commands))


if __name__ == "__main__":
    main()
