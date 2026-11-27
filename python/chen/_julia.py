"""The Julia runtime behind chen, started on the first call that needs it.

`import chen` does not start Julia: the first computation does. chen starts libjulia
itself (through ctypes, from the Julia environment juliapkg manages), loads
ChenSignatures and calls its C entry points (`ChenSignatures.PyAPI`) directly: no
PythonCall, and no JIT compilation, since the entry points are precompiled. About
0.8 s from `import chen` to the first result.

If juliacall is already running in the process, its Julia runtime is reused.
"""
import atexit
import ctypes as c
import os
import signal
import sys
import threading
from pathlib import Path

_UUID = "4efb4129-5e83-47d2-926d-947c0e6cb76d"
_CUDA = ("CUDA", "052768ef-5323-5732-b1bb-66c8b64840ba")

_lock = threading.RLock()
_lib = None      # libjulia (ctypes)
_api = None      # CPU entry points
_gpu = None      # CUDA entry points

I, P, D = c.c_int64, c.c_void_p, c.c_double
PI = c.POINTER(c.c_int64)

# name -> (restype, argtypes), in the order of ChenSignatures.PyAPI.CNAMES
_SIGNATURES = {
    "sig":           (c.c_int, [I, P, P, I, I, I, I, I]),
    "prepare":       (c.c_int, [I, I, PI]),
    "logsig_length": (c.c_int, [I, I, PI]),
    "logsig":        (c.c_int, [I, P, P, I, I, I, I, I, I]),
    "sig_gram":      (c.c_int, [I, P, P, I, I, P, I, I, I, I, I, I, D, I, I]),
    "sig_fwd":       (c.c_int, [I, P, P, I, I, I, I, I, PI]),
    "logsig_fwd":    (c.c_int, [I, P, P, I, I, I, I, I, I, PI]),
    "sig_gram_fwd":  (c.c_int, [I, P, P, I, I, P, I, I, I, I, I, I, D, I, PI]),
    "pull":          (c.c_int, [I, P, P, P]),
    "release":       (c.c_int, [I]),
    "nthreads":      (I, []),
    "lasterror":     (I, [c.c_char_p, I]),
}
_GPU_SIGNATURES = {
    "sig_gram":      (c.c_int, [I, P, P, I, I, P, I, I, I, I, I, I, D]),
    "sig_gram_fwd":  (c.c_int, [I, P, P, I, I, P, I, I, I, I, I, I, D, PI]),
    "pull":          (c.c_int, [I, P, P, P]),
}


class JuliaError(RuntimeError):
    """An exception raised in Julia."""


def _repo_root():
    """The Julia package checkout when running from the repository, else None."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "Project.toml").exists() and (root / "src" / "ChenSignatures.jl").exists() else None


def _setup_package():
    import juliapkg
    from chen._version import __version__

    root = _repo_root()
    if root is not None:
        juliapkg.add("ChenSignatures", uuid=_UUID, path=str(root), dev=True)
    else:
        juliapkg.add("ChenSignatures", uuid=_UUID, version=f"={__version__}")


def _threads():
    for var in ("CHEN_JULIA_THREADS", "PYTHON_JULIACALL_THREADS", "JULIA_NUM_THREADS"):
        if os.environ.get(var):
            return os.environ[var]
    return "auto"


def _start_julia():
    """Initialise libjulia in this process; return the ctypes handle."""
    jc = sys.modules.get("juliacall")
    if jc is not None and jc.CONFIG.get("inited"):
        return jc.CONFIG["lib"]                      # juliacall already runs Julia
    import juliapkg

    _setup_package()
    juliapkg.resolve()
    exe, project = juliapkg.executable(), juliapkg.project()
    libpath, bindir = juliapkg.libjulia(), juliapkg.bindir()
    lib = c.PyDLL(libpath, mode=c.RTLD_GLOBAL)
    argv = [exe, f"--project={project}", f"--home={bindir}", f"--threads={_threads()}",
            "--handle-signals=yes", "--startup-file=no"]
    argv = [a.encode() for a in argv]
    argc = c.c_int(len(argv))
    argvp = c.POINTER(c.c_char_p)((c.c_char_p * len(argv))(*argv))
    lib.jl_parse_opts.argtypes = [c.c_void_p, c.c_void_p]
    lib.jl_parse_opts.restype = None
    py_sigint = signal.getsignal(signal.SIGINT)
    lib.jl_parse_opts(c.pointer(argc), c.pointer(argvp))
    try:
        init = lib.jl_init_with_image__threading
    except AttributeError:
        init = lib.jl_init_with_image
    init.argtypes = [c.c_char_p, c.c_char_p]
    init.restype = None
    init(bindir.encode(), None)
    # Julia needs its handlers for the GC (SIGSEGV, SIGUSR2, ...), not for SIGINT: give
    # Ctrl-C back to Python (a running Julia call finishes first).
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, py_sigint if py_sigint is not None else signal.default_int_handler)
    lib.jl_atexit_hook.argtypes = [c.c_int]
    lib.jl_atexit_hook.restype = None
    atexit.register(_shutdown, lib)
    # Julia can be initialised once per process: stop a later `import juliacall` from
    # initialising it again (it raises instead of crashing). To use juliacall together
    # with chen, import juliacall before chen's first computation.
    os.environ["PYTHON_JULIACALL_INIT"] = "no"
    return lib


def _shutdown(lib):
    """Stop Julia at exit. Disable the entry points first: Handles that Python
    finalises later (bases, pullbacks still alive at shutdown) must not call into a
    runtime that is gone."""
    global _api, _gpu
    _api = _gpu = None
    lib.jl_atexit_hook(0)


def _eval(lib, code):
    lib.jl_eval_string.argtypes = [c.c_char_p]
    lib.jl_eval_string.restype = c.c_void_p
    lib.jl_exception_occurred.restype = c.c_void_p
    v = lib.jl_eval_string(code.encode())
    if not v or lib.jl_exception_occurred():
        lib.jl_exception_clear()
        raise JuliaError(f"Julia failed to evaluate: {code}")
    return v


def _bind_table(lib, table_expr, signatures):
    """Read a (address, length) pointer table from Julia and wrap each entry."""
    lib.jl_unbox_int64.argtypes = [c.c_void_p]
    lib.jl_unbox_int64.restype = c.c_int64
    lib.jl_get_nth_field.argtypes = [c.c_void_p, c.c_size_t]
    lib.jl_get_nth_field.restype = c.c_void_p
    t = _eval(lib, table_expr)
    addr = lib.jl_unbox_int64(lib.jl_get_nth_field(t, 0))
    n = lib.jl_unbox_int64(lib.jl_get_nth_field(t, 1))
    if n != len(signatures):
        raise JuliaError(f"ChenSignatures exposes {n} entry points, chen expects {len(signatures)}: "
                         "the Julia and Python packages are out of sync")
    ptrs = (c.c_void_p * n).from_address(addr)
    fns = {}
    for (name, (res, args)), p in zip(signatures.items(), ptrs):
        fns[name] = c.CFUNCTYPE(res, *args)(p)     # CFUNCTYPE: the GIL is released during calls
    return fns


class _API:
    """The C entry points, raising JuliaError on failure."""

    def __init__(self, fns, lasterror):
        self._fns = fns
        self._lasterror = lasterror

    def __getattr__(self, name):
        f = self._fns[name]
        if f.restype is not c.c_int:
            return f

        def call(*args):
            if f(*args) != 0:
                raise JuliaError(self._lasterror())
        call.__name__ = name
        return call


def _lasterror_reader(f):
    def read():
        n = f(None, 0)
        buf = c.create_string_buffer(n)
        f(buf, n)
        return buf.raw.decode("utf-8", "replace")
    return read


def api():
    """ChenSignatures.PyAPI's C entry points, starting Julia if needed."""
    global _lib, _api
    if _api is None:
        with _lock:
            if _api is None:
                lib = _start_julia()
                _eval(lib, "using ChenSignatures")
                fns = _bind_table(lib, "ChenSignatures.PyAPI.cpointers_table()", _SIGNATURES)
                _lib = lib
                _api = _API(fns, _lasterror_reader(fns["lasterror"]))
    return _api


def julia_eval(code):
    """Evaluate Julia code for its side effects (testing and diagnostics)."""
    api()
    _eval(_lib, code)


class Handle:
    """A Julia object kept alive in ChenSignatures.PyAPI's registry; released on GC."""

    __slots__ = ("h", "__weakref__")

    def __init__(self, h):
        self.h = h

    def __del__(self):
        a = _api
        if a is not None and self.h:
            try:
                a.release(self.h)
            except Exception:
                pass
            self.h = 0


def new_handle(f, *args):
    """Call an entry point that stores a handle in its last argument."""
    h = c.c_int64(0)
    f(*args, c.byref(h))
    return Handle(h.value)


def install_gpu():
    """Add CUDA.jl to chen's Julia environment (once; a large download).

    After this, `chen.torch` functions given CUDA tensors run on the GPU.
    """
    import juliapkg

    juliapkg.add(_CUDA[0], uuid=_CUDA[1])
    juliapkg.resolve()


def gpu_api():
    """The CUDA entry points (chen/gpu.jl), loading CUDA.jl on first use."""
    global _gpu
    if _gpu is None:
        a = api()
        with _lock:
            if _gpu is None:
                try:
                    _eval(_lib, "using CUDA")
                except JuliaError as e:
                    raise RuntimeError(
                        "GPU support needs CUDA.jl in chen's Julia environment: "
                        "run `chen.install_gpu()` once (then restart Python)."
                    ) from e
                path = str(Path(__file__).with_name("gpu.jl")).replace("\\", "\\\\")
                _eval(_lib, f'include(raw"{path}")')
                fns = _bind_table(_lib, "ChenPyGPU.cpointers_table()", _GPU_SIGNATURES)
                # torch runs CUDA backward passes on per-device worker threads; Julia
                # code that yields to its task scheduler (CUDA.jl does) deadlocks there.
                # Run backward passes on the calling thread instead (a global torch
                # setting; it only serializes backward passes across several GPUs).
                import torch

                torch.autograd.set_multithreading_enabled(False)
                _gpu = _API(fns, a._lasterror)
    return _gpu
