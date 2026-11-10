"""Signed 64-bit integer conversion shared by analysis and execution."""


def to_i64(value: int) -> int:
    """Wrap an integer to the signed 64-bit range."""
    return (value + (1 << 63)) % (1 << 64) - (1 << 63)
