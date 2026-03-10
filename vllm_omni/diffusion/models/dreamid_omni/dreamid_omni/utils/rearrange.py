from einops import rearrange


class Rearrange:
    def __init__(self, pattern: str, **kwargs):
        self.pattern = pattern
        self.kwargs = kwargs

    def __call__(self, x):
        return rearrange(x, self.pattern, **self.kwargs)

