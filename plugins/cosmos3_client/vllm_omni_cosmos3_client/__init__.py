"""Optional client plugin; importing it never imports the GPU server stack."""


def main() -> None:
    from .closed_loop_eval import main as evaluate

    evaluate()
