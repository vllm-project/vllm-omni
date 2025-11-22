import zmq
import psutil
import socket
import tempfile

from vllm.logger import init_logger

logger = init_logger(__name__)

def get_zmq_socket(
    context: zmq.Context,
    socket_type: zmq.SocketType,
    endpoint: str,
    bind: bool,
    max_bind_retries: int = 10,
) -> tuple[zmq.Socket, str]:
    """
    Create and configure a ZMQ socket.

    Args:
        context: ZMQ context
        socket_type: Type of ZMQ socket
        endpoint: Endpoint string (e.g., "tcp://localhost:5555")
        bind: Whether to bind (True) or connect (False)
        max_bind_retries: Maximum number of retries if bind fails due to address already in use

    Returns:
        A tuple of (socket, actual_endpoint). The actual_endpoint may differ from the
        requested endpoint if bind retry was needed.
    """
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)
    else:
        buf_size = -1

    socket = context.socket(socket_type)
    if endpoint.find("[") != -1:
        socket.setsockopt(zmq.IPV6, 1)

    def set_send_opt():
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    def set_recv_opt():
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type == zmq.PUSH:
        set_send_opt()
    elif socket_type == zmq.PULL:
        set_recv_opt()
    elif socket_type == zmq.DEALER:
        set_send_opt()
        set_recv_opt()
    elif socket_type == zmq.REQ:
        set_send_opt()
        set_recv_opt()
    elif socket_type == zmq.REP:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")

    if bind:
        # Parse port from endpoint for retry logic
        import re

        port_match = re.search(r":(\d+)$", endpoint)

        if port_match and max_bind_retries > 1:
            original_port = int(port_match.group(1))
            last_exception = None

            for attempt in range(max_bind_retries):
                try:
                    current_endpoint = endpoint
                    if attempt > 0:
                        # Try next port (increment by 42 to match settle_port logic)
                        current_port = original_port + attempt * 42
                        current_endpoint = re.sub(
                            r":(\d+)$", f":{current_port}", endpoint
                        )
                        logger.info(
                            f"ZMQ bind failed for port {original_port + (attempt - 1) * 42}, "
                            f"retrying with port {current_port} (attempt {attempt + 1}/{max_bind_retries})"
                        )

                    socket.bind(current_endpoint)

                    if attempt > 0:
                        logger.warning(
                            f"Successfully bound ZMQ socket to {current_endpoint} after {attempt + 1} attempts. "
                            f"Original port {original_port} was unavailable."
                        )

                    return socket, current_endpoint

                except zmq.ZMQError as e:
                    last_exception = e
                    if e.errno == zmq.EADDRINUSE and attempt < max_bind_retries - 1:
                        # Address already in use, try next port
                        continue
                    elif attempt == max_bind_retries - 1:
                        # Last attempt failed
                        logger.error(
                            f"Failed to bind ZMQ socket after {max_bind_retries} attempts. "
                            f"Original endpoint: {endpoint}, Last tried port: {original_port + attempt * 42}"
                        )
                        raise
                    else:
                        # Different error, raise immediately
                        raise

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
        else:
            # No retry logic needed (either no port in endpoint or max_bind_retries == 1)
            socket.bind(endpoint)
            return socket, endpoint
    else:
        socket.connect(endpoint)
        return socket, endpoint

    return socket, endpoint

def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except socket.error:
            return False
        except OverflowError:
            return False