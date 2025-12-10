import hashlib
from typing import Union

def generate_md5(data: Union[str, bytes]) -> str:
    """
    Return hex MD5 digest for input str or bytes.
    Accepts text (utf-8) or bytes.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    h = hashlib.md5()
    h.update(data)
    return h.hexdigest()

# If the module already defines an equivalent helper with another name,
# prefer that implementation (keeps backward-compatibility).
# For example, if there's md5_hex(), use it as the implementation.
if "md5_hex" in globals() and callable(globals()["md5_hex"]):
    generate_md5 = globals()["md5_hex"]

# Export symbol for from ... import generate_md5 usage
if "__all__" in globals():
    if "generate_md5" not in globals()["__all__"]:
        globals()["__all__"].append("generate_md5")
else:
    __all__ = ["generate_md5"]