class InvalidInputReferenceError(ValueError):
    def __init__(self, message: str = "Invalid input reference.") -> None:
        super().__init__(message)
