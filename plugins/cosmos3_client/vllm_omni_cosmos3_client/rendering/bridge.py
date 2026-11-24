"""Expose render-only reverse forwards to Huawei Docker containers."""

import argparse
import select
import socket
import socketserver
import threading


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(listen, upstream_port):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with socket.create_connection(("127.0.0.1", upstream_port), timeout=30) as upstream:
                upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while True:
                    ready, _, _ = select.select([self.request, upstream], [], [], 300)
                    if not ready:
                        return
                    for source in ready:
                        data = source.recv(262144)
                        if not data:
                            return
                        (upstream if source is self.request else self.request).sendall(data)

    with Server(("172.17.0.1", listen), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", action="append", required=True, help="bridge_port:loopback_port")
    args = parser.parse_args()
    threads = []
    for mapping in args.mapping:
        listen, upstream = map(int, mapping.split(":"))
        thread = threading.Thread(target=serve, args=(listen, upstream))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
