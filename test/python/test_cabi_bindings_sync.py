"""Drift guard for the ctypes bindings of libdfkv.

The HiCache plugin, the vLLM connector, the LMCache connector and
dfkv_common each declare argtypes/restype for the native C ABI. Nothing
kept them in step with each other or with src/client/dfkv_c_api.h: a
changed prototype could be fixed in one binding and left wrong in another,
and ctypes would pass the wrong types without an error. This test loads
every binding against a recording stand-in for the library and checks that
each symbol bound in more than one place is bound the same way, and that
its argument count is the header's.
"""
import ctypes
import importlib
import os
import re
import sys
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HEADER = os.path.join(ROOT, "src", "client", "dfkv_c_api.h")


class _Fn:
    def __init__(self):
        self.argtypes = None
        self.restype = ctypes.c_int


class _RecordingLib:
    """Every attribute is a function object whose argtypes/restype are kept."""

    def __init__(self, *_args, **_kwargs):
        object.__setattr__(self, "fns", {})

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self.fns.setdefault(name, _Fn())


def _signature(fn):
    def name(t):
        return getattr(t, "__name__", repr(t))
    return (tuple(name(t) for t in fn.argtypes or ()), name(fn.restype))


def _bindings():
    """{module: {symbol: (argtypes, restype)}} for every binding site."""
    sys.path[:0] = [
        HERE,
        os.path.join(ROOT, "integration", "hicache"),
        os.path.join(ROOT, "integration", "common", "src"),
        os.path.join(ROOT, "integration", "vllm", "src"),
        os.path.join(ROOT, "integration", "lmcache", "src"),
    ]
    sites = {
        "hicache": ("dfkv_hicache", "_load_lib"),
        "vllm": ("dfkv_vllm._cabi", "load_lib"),
        "lmcache": ("dfkv_connector.native_client", "load_lib"),
    }
    # The connector packages' __init__ imports their framework (vLLM,
    # LMCache); the binding modules themselves do not. Import them without it.
    for package, path in (
            ("dfkv_vllm", os.path.join(ROOT, "integration", "vllm", "src",
                                       "dfkv_vllm")),
            ("dfkv_connector", os.path.join(ROOT, "integration", "lmcache",
                                            "src", "dfkv_connector"))):
        if package not in sys.modules:
            stub = types.ModuleType(package)
            stub.__path__ = [path]
            sys.modules[package] = stub
    out = {}
    with mock.patch.object(ctypes, "CDLL", _RecordingLib):
        for site, (module, loader) in sites.items():
            mod = importlib.import_module(module)
            lib = getattr(mod, loader)("libdfkv-recording.so")
            from dfkv_common import cabi
            cabi.bind_retain(lib)
            out[site] = {name: _signature(fn) for name, fn in lib.fns.items()
                         if fn.argtypes is not None}
    return out


def _header_arity():
    with open(HEADER) as header:
        text = header.read()
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    arity = {}
    for m in re.finditer(r"\b(dfkv_\w+)\s*\(([^;{]*?)\)\s*;", text):
        params = m.group(2).strip()
        arity[m.group(1)] = 0 if params in ("", "void") else params.count(",") + 1
    return arity


class CabiBindingsAgree(unittest.TestCase):
    def test_bindings_agree_with_each_other_and_the_header(self):
        sites = _bindings()
        self.assertEqual(set(sites), {"hicache", "vllm", "lmcache"})
        arity = _header_arity()
        seen = {}
        problems = []
        for site, symbols in sites.items():
            for name, sig in symbols.items():
                if name not in arity:
                    problems.append(f"{site}: {name} is not in dfkv_c_api.h")
                elif len(sig[0]) != arity[name]:
                    problems.append(
                        f"{site}: {name} binds {len(sig[0])} args, the header "
                        f"declares {arity[name]}")
                first = seen.setdefault(name, (site, sig))
                if first[1] != sig:
                    problems.append(
                        f"{name}: {first[0]} binds {first[1]}, {site} {sig}")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
