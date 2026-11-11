"""Fuzz driver that runs inside the embedded interpreter of mayhem/fuzz_nanobind.cpp.

The fuzzer bytes are decoded into a short program: a few calls into the functions and classes that
nanobind's own test extensions bind (tests/test_*.cpp), with Python objects built from the bytes as
arguments. Python-level exceptions are the normal answer of a bound function to a bad argument and are
swallowed; only a crash, sanitizer report or hang inside nanobind / the bound C++ is a finding.
"""
import array
import datetime
import gc
import importlib
import itertools
import os
import sys

import nbfuzz_native as native

try:
    import numpy as np
except Exception:
    np = None

# Group name -> test extensions (built from tests/test_<name>.cpp by mayhem/build.sh).
GROUPS = {
    "functions": ["functions", "accessor", "callbacks", "exception", "enum", "make_iterator",
                  "issue", "chrono", "classes", "holders", "intrusive"],
    "stl": ["stl", "stl_bind_map", "stl_bind_vector"],
    "ndarray": ["ndarray", "jax", "tensorflow", "eigen"],
}

# Functions that spawn threads or release the GIL on purpose add nothing for a byte-driven fuzzer.
SKIP_NAMES = ("gil", "thread")

# Test fixtures that are not nanobind code paths: they allocate as many elements as an int argument says, or
# index the array they receive at hard-coded positions, so a short array is an out-of-bounds read in the
# fixture itself. Matched as substrings of "<module>.<name>".
SKIP_FIXTURES = (
    "_in_ptr", "query_shared", "stored_shared_value", "variant_nondefault", "castTo",
    "updateRefVXi", "array_in", "type_dict", "type_lookup", "freeze_type", "inst_dict", "bytearray_resize", "bytearray_new", "get_vnc", "test_ndarray_ext.get_stride", "cnt_throw_after", "Bench",
    "test_ndarray_ext.fill_view", "test_ndarray_ext.check_ro_by", "test_ndarray_ext.check_rw_by",
    "test_ndarray_ext.inspect_", "test_ndarray_ext.set_item", "test_ndarray_ext.initialize",
    "test_ndarray_ext.process", "test_ndarray_ext.check_shape_ptr", "test_ndarray_ext.check_stride_ptr",
)

# Fixtures that crash in the test's own code for an arbitrary argument, matched as the exact
# "<module>.<name>" (a substring would also hit "get_destructed", "Dog", ...):
#   owns_cpp: nb::inst_state() on a handle that is not an instance reads out of bounds;
#   test_list: l[2] = 123 on a list shorter than 3 (unchecked PyList_SET_ITEM in the fixture);
#   test_18: nb::bytes(c, size) with an arbitrary int size;
#   D and the get_d* functions: D(int) computes d + 10000 and overflows (signed, UBSan);
#   returnVecXd: builds an Eigen vector from a caller-controlled size and trips an Eigen assert.
# The dynamic-size Eigen binary fixtures (see _eigen_dynamic_binary) are skipped too.
SKIP_EXACT = (
    "test_holders_ext.owns_cpp", "test_functions_ext.test_list", "test_functions_ext.test_18",
    "test_classes_ext.D", "test_classes_ext.get_d", "test_classes_ext.get_optional_d",
    "test_classes_ext.get_d_via_cast", "test_classes_ext.get_d_via_try_cast",
    "test_eigen_ext.returnVecXd",
)
SKIP_METHODS = ("returnVecXd",)


def _eigen_dynamic_binary(modname, n):
    # test_eigen.cpp's add* fixtures compute `a + b` on two caller-supplied operands with no shape check, so
    # for a dynamic-size type (VectorXi, MatrixXu*, Ref/Map/DRef of those) two different shapes trip Eigen's
    # own assert in the fixture. The fixed-size ones (V3i, R3i, A3i, M4u*) are fine: the caster enforces the shape.
    return modname == "test_eigen_ext" and n.startswith("add") and ("VX" in n or "MX" in n or "DRef" in n)

# Fixture state that outlives an input and has a Python-visible reset, drained after every input.
DRAINS = (
    ("test_functions_ext", "call_policy_record"),
    ("test_classes_ext", "get_destructed"),
    ("test_classes_ext", "pooled_reset"),
    ("test_classes_ext", "pooled_gc_reset"),
    ("test_classes_ext", "pooled_tr_reset"),
    ("test_holders_ext", "clear_stored_shared"),
    ("test_holders_ext", "reset"),
)
DRAIN_FUNCS = []
STATIC_PROPS = None

FUNCS = []
CLASSES = []

_devnull = open(os.devnull, "w")


class Idx:
    def __init__(self, v):
        self.v = v

    def __index__(self):
        return self.v

    def __int__(self):
        return self.v

    def __float__(self):
        return float(self.v)

    def __len__(self):
        return self.v


class Seq:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class Raiser:
    def __iter__(self):
        raise RuntimeError("iter")

    def __len__(self):
        raise RuntimeError("len")

    def __getitem__(self, i):
        raise RuntimeError("getitem")

    def __index__(self):
        raise RuntimeError("index")

    def __float__(self):
        raise RuntimeError("float")

    def __bool__(self):
        raise RuntimeError("bool")

    def __hash__(self):
        raise RuntimeError("hash")


class ShrinkingList(list):
    def __iter__(self):
        for i, x in enumerate(list(self)):
            self.clear()
            yield x


class StrSub(str):
    pass


class IntSub(int):
    pass


class FloatSub(float):
    pass


class ListSub(list):
    pass


class TupleSub(tuple):
    pass


class DictSub(dict):
    pass


class BytesSub(bytes):
    pass


class BufObj:
    def __init__(self, mv):
        self.mv = mv

    def __buffer__(self, flags):
        return memoryview(self.mv)


def mkgen(items):
    yield from items


def _mut(*a, **k):
    for x in a:
        if isinstance(x, (list, dict, set, bytearray)):
            x.clear()
    return a


def _raise(*a, **k):
    raise ValueError("cb")


CALLBACK_SPECS = [
    lambda *a, **k: a,
    lambda *a, **k: None,
    lambda *a, **k: 1,
    lambda *a, **k: "x",
    lambda *a, **k: [a],
    lambda *a, **k: mkgen(a),
    _mut,
    _raise,
    int, str, list, dict, len, bool, float, tuple, set, repr, iter, type,
]



def _subclassable(t):
    try:
        type("T", (t,), {})
        return True
    except TypeError:
        return False


# Only subclassable types: each input gets fresh subclasses (below), because a builtin type used as the
# `nurse` of keep_alive keeps its weakref callback and patient alive for the whole process.
TYPE_SPECS = [t for t in (int, str, bytes, list, tuple, dict, set, frozenset, float, bool, complex, object,
                          type, memoryview, bytearray, type(None), datetime.datetime) if _subclassable(t)]

_inst = {}


def _wrap(f):
    return lambda *a, **k: f(*a, **k)


def pick_callback(r):
    i = r.u8() % len(CALLBACK_SPECS)
    if ("cb", i) not in _inst:
        _inst[("cb", i)] = _wrap(CALLBACK_SPECS[i])
    return _inst[("cb", i)]


def pick_type(r):
    i = r.u8() % len(TYPE_SPECS)
    if ("ty", i) not in _inst:
        _inst[("ty", i)] = type("T", (TYPE_SPECS[i],), {})
    return _inst[("ty", i)]

SPECIAL_FLOATS = [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1e308, 5e-324, 0.5, 1.5,
                  2.0 ** 31, 2.0 ** 63, -2.0 ** 63, 2.0 ** 64, 1e-7]
SPECIAL_INTS = [0, 1, -1, 127, 128, 255, 256, 2 ** 15, 2 ** 31 - 1, 2 ** 31, -2 ** 31, 2 ** 32,
                2 ** 63 - 1, 2 ** 63, -2 ** 63, 2 ** 64 - 1, 2 ** 64, 2 ** 100]
NP_DTYPES = ["?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "e", "f", "d", "F", "D", "S3", "U2",
             "V4", "M8[s]", "m8[ns]", "<i4", ">i4", ">f8", "O"]
ARRAY_CODES = "bBhHiIlLqQfd"
FORMATS = [None, b"B", b"b", b"c", b"?", b"h", b"H", b"i", b"I", b"l", b"L", b"q", b"Q", b"e", b"f", b"d",
           b"Zf", b"Zd", b"<i", b">i", b"=i", b"@i", b"!i", b"2i", b"T{i}", b"", b"\x00x", b"x", b"s",
           b"10s", b"O", b"P", b"n", b"N", b"F", b"D", b"<", b">", b"@", b"=", b"Zi"]
DL_DTYPES = [(0, 8), (0, 16), (0, 32), (0, 64), (1, 8), (1, 16), (1, 32), (1, 64), (2, 16), (2, 32), (2, 64),
             (4, 16), (5, 64), (5, 128), (6, 8), (7, 8), (8, 8), (12, 8), (15, 8), (17, 8), (18, 32),
             (0, 1), (1, 4), (0, 3), (3, 8), (0, 24), (2, 8)]


class Reader:
    __slots__ = ("d", "i")

    def __init__(self, d):
        self.d = d
        self.i = 0

    def more(self):
        return self.i < len(self.d)

    def u8(self):
        i = self.i
        self.i = i + 1
        return self.d[i] if i < len(self.d) else 0

    def take(self, n):
        b = self.d[self.i:self.i + n]
        self.i += n
        return bytes(b)

    def int64(self):
        return int.from_bytes(self.take(8).ljust(8, b"\0"), "little", signed=True)

    def idx(self, seq):
        return seq[self.u8() % len(seq)]


def gen_scalar(r):
    k = r.u8() % 9
    if k == 0:
        return None
    if k == 1:
        return bool(r.u8() & 1)
    if k == 2:
        return r.u8() - 128
    if k == 3:
        return r.int64()
    if k == 4:
        return r.idx(SPECIAL_INTS)
    if k == 5:
        return r.idx(SPECIAL_FLOATS)
    if k == 6:
        return gen_str(r)
    if k == 7:
        return r.take(r.u8() % 12)
    return str(r.int64())


def gen_str(r):
    raw = r.take(r.u8() % 24)
    try:
        return raw.decode("utf-8", "surrogatepass")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def gen_buffer(r):
    data = r.take(r.u8() % 48)
    fmt = r.idx(FORMATS)
    if r.u8() & 3 == 0:
        fmt = r.take(r.u8() % 6)
    itemsize = r.idx([1, 1, 2, 4, 8, 16, 3, 32])
    ndim = r.u8() % 4
    shape = tuple(r.u8() % 6 for _ in range(ndim))
    return native.buffer(data, fmt, itemsize, shape, r.u8() % 3, r.u8() & 1)


def gen_dlpack(r, pool):
    data = r.take(r.u8() % 48)
    code, bits = r.idx(DL_DTYPES)
    lanes = 1 if r.u8() & 3 else r.idx([1, 2, 4, 0])
    ndim = r.u8() % 4
    shape = tuple(r.u8() % 6 for _ in range(ndim))
    return native.dlpack(data, code, bits, lanes, shape, r.u8() % 3, (r.u8() % 3) * 8,
                         r.idx([1, 1, 1, 3, 11]), r.u8() & 1)


def gen_numpy(r):
    raw = bytearray(r.take(r.u8() % 64))
    dt = np.dtype(r.idx(NP_DTYPES))
    if dt.itemsize == 0:
        dt = np.dtype("B")
    raw = raw[:len(raw) - len(raw) % dt.itemsize]
    if dt.kind == "O":
        a = np.array([None] * (len(raw) // 8), dtype=object)
    else:
        a = np.frombuffer(raw, dtype=dt)
    shp = r.u8() % 5
    if a.size and shp == 1:
        a = a.reshape(a.size, 1)
    elif a.size and shp == 2:
        for d in (2, 3, 4):
            if a.size % d == 0:
                a = a.reshape(d, a.size // d)
                break
    elif shp == 3:
        a = a.reshape(())if a.size == 1 else a
    sel = r.u8() % 8
    if sel == 1:
        a = a.T
    elif sel == 2:
        a = a[::2]
    elif sel == 3:
        a = a[::-1]
    elif sel == 4:
        a = np.asfortranarray(a)
    elif sel == 5 and a.ndim:
        a = a[:0]
    elif sel == 6:
        a = np.array(a)
    elif sel == 7:
        a = a.view(np.uint8)
    if r.u8() & 1:
        a.flags.writeable = False
    return a


def gen_exotic(r, depth, pool):
    k = r.u8() % 24
    if k == 0:
        return Idx(r.int64() if r.u8() & 1 else r.u8())
    if k == 1:
        return Seq([gen(r, depth + 1, pool) for _ in range(r.u8() % 4)])
    if k == 2:
        return Raiser()
    if k == 3:
        return ShrinkingList([gen(r, depth + 1, pool) for _ in range(r.u8() % 4)])
    if k == 4:
        return StrSub(gen_str(r))
    if k == 5:
        return IntSub(r.int64())
    if k == 6:
        return FloatSub(r.idx(SPECIAL_FLOATS))
    if k == 7:
        return ListSub(gen(r, depth + 1, pool) for _ in range(r.u8() % 4))
    if k == 8:
        return TupleSub(gen(r, depth + 1, pool) for _ in range(r.u8() % 4))
    if k == 9:
        return DictSub({gen_scalar(r): gen(r, depth + 1, pool) for _ in range(r.u8() % 3)})
    if k == 10:
        return BytesSub(r.take(r.u8() % 12))
    if k == 11:
        return BufObj(memoryview(r.take(r.u8() % 32)))
    if k == 12:
        return pick_callback(r)
    if k == 13:
        return pick_type(r)
    if k == 14:
        return mkgen([gen(r, depth + 1, pool) for _ in range(r.u8() % 5)])
    if k == 15:
        return iter([gen(r, depth + 1, pool) for _ in range(r.u8() % 5)])
    if k == 16:
        return slice(r.idx([None, 0, 1, -1, 2 ** 70]), r.idx([None, 3, -2, 2 ** 70]), r.idx([None, 1, -1, 0]))
    if k == 17:
        return complex(r.idx(SPECIAL_FLOATS), r.idx(SPECIAL_FLOATS))
    if k == 18:
        return range(r.int64() % 1000, r.int64() % 1000, r.idx([1, -1, 3]))
    if k == 19:
        return r.idx([Ellipsis, NotImplemented, object(), type, object])
    if k == 20:
        return gen_chrono(r)
    if k == 21:
        return memoryview(r.take(r.u8() % 32))
    if k == 22:
        return array.array(r.idx(list(ARRAY_CODES)),
                           bytes(r.take(r.u8() % 32)).ljust(32, b"\0")[:32])
    return object()


def gen_chrono(r):
    k = r.u8() % 6
    try:
        if k == 0:
            return datetime.timedelta(days=r.int64() % 999999999, seconds=r.int64() % 86400,
                                      microseconds=r.int64() % 10 ** 6)
        if k == 1:
            return datetime.datetime(1 + r.u8() * 39 % 9999, 1 + r.u8() % 12, 1 + r.u8() % 28, r.u8() % 24,
                                     r.u8() % 60, r.u8() % 60, r.int64() % 10 ** 6)
        if k == 2:
            return datetime.date(1 + r.u8() * 39 % 9999, 1 + r.u8() % 12, 1 + r.u8() % 28)
        if k == 3:
            return datetime.time(r.u8() % 24, r.u8() % 60, r.u8() % 60, r.int64() % 10 ** 6)
        if k == 4:
            return datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(minutes=r.u8() - 128)))
        return datetime.timedelta(microseconds=r.int64())
    except Exception:
        return None


def gen(r, depth=0, pool=None):
    k = r.u8() % 30
    if depth >= 3 and k >= 9:
        k %= 9
    if k < 9:
        return gen_scalar(r)
    if k == 9:
        return [gen(r, depth + 1, pool) for _ in range(r.u8() % 6)]
    if k == 10:
        return tuple(gen(r, depth + 1, pool) for _ in range(r.u8() % 6))
    if k == 11:
        return {gen_scalar(r): gen(r, depth + 1, pool) for _ in range(r.u8() % 4)}
    if k == 12:
        return {gen_scalar(r) for _ in range(r.u8() % 5)}
    if k == 13:
        return frozenset(gen_scalar(r) for _ in range(r.u8() % 5))
    if k == 14:
        return bytearray(r.take(r.u8() % 24))
    if k == 15 or k == 16:
        return gen_buffer(r)
    if k == 17:
        return gen_dlpack(r, pool)
    if k == 18 and np is not None:
        return gen_numpy(r)
    if k == 19 and pool:
        return r.idx(pool)
    if k == 20:
        return array.array(r.idx(list(ARRAY_CODES)), bytes(r.take(r.u8() % 32)).ljust(32, b"\0")[:32])
    if k == 21:
        return memoryview(r.take(r.u8() % 32))
    return gen_exotic(r, depth, pool)


NO_TOP_NONE = False


def gen_args(r, pool):
    args = [gen(r, 0, pool) for _ in range(r.u8() % 5)]
    if NO_TOP_NONE:
        # A bare None reaches the raw-pointer parameters of the fixtures, which dereference them.
        args = [0 if a is None else a for a in args]
    return args


def gen_kwargs(r, pool):
    if r.u8() & 3:
        return {}
    return {r.idx(["a", "b", "x", "y", "key", "value", "n", "i", "j", "k", "self", "up", "down", "p"]):
            gen(r, 0, pool) for _ in range(r.u8() % 3)}


def after(res, pool):
    if res is None:
        return
    if len(pool) < 16:
        pool.append(res)
    repr(res)
    if not isinstance(res, (str, bytes, memoryview)):
        try:
            list(itertools.islice(iter(res), 8))
        except Exception:
            pass


def op(r, pool):
    sel = r.u8()
    k = sel % 8
    b = r.u8() | (r.u8() << 8)
    c = r.u8() | (r.u8() << 8)
    args = gen_args(r, pool)
    kwargs = gen_kwargs(r, pool)
    if k < 4 and FUNCS:
        after(FUNCS[b % len(FUNCS)](*args, **kwargs), pool)
        return
    if not CLASSES:
        return
    cls, names = CLASSES[b % len(CLASSES)]
    if k < 5:
        after(cls(*args, **kwargs), pool)
        return
    obj = cls(*args, **kwargs)
    pool.append(obj)
    if not names:
        return
    name = names[c % len(names)]
    attr = getattr(obj, name)
    if k == 7:
        setattr(obj, name, gen(r, 0, pool))
        return
    if callable(attr):
        after(attr(*gen_args(r, pool)), pool)


def init(group):
    global NO_TOP_NONE, STATIC_PROPS
    NO_TOP_NONE = group != "ndarray"
    sys.stdout = _devnull
    sys.stderr = _devnull
    for name in GROUPS[group]:
        modname = "test_%s_ext" % name
        try:
            mod = importlib.import_module(modname)
        except ImportError as e:
            if name == "eigen":
                continue
            sys.__stderr__.write("nbfuzz: cannot import %s: %s\n" % (modname, e))
            raise
        for n in sorted(dir(mod)):
            if n.startswith("__") or any(s in n for s in SKIP_NAMES) \
                    or any(s in modname + "." + n for s in SKIP_FIXTURES) \
                    or modname + "." + n in SKIP_EXACT or _eigen_dynamic_binary(modname, n):
                continue
            o = getattr(mod, n)
            if isinstance(o, type):
                if getattr(o, "__module__", None) != modname:
                    continue
                names = sorted({a for c in o.__mro__ if c is not object for a in vars(c)
                                if a not in ("__module__", "__doc__", "__dict__", "__weakref__",
                                             "__qualname__", "__init__", "__new__", "__del__")})
                CLASSES.append((o, [a for a in names
                                   if not any(s in a for s in SKIP_NAMES) and a not in SKIP_METHODS]))
            elif callable(o):
                FUNCS.append(o)
    if not FUNCS and not CLASSES:
        raise RuntimeError("nbfuzz: no bound callables found for group " + group)
    for modname, n in DRAINS:
        mod = sys.modules.get(modname)
        if mod is not None and hasattr(mod, n):
            DRAIN_FUNCS.append(getattr(mod, n))
    cls = getattr(sys.modules.get("test_classes_ext"), "StaticProperties", None)
    if cls is not None:
        STATIC_PROPS = (cls, cls.value)
    # Startup objects are never garbage: keep them out of the per-input collections.
    gc.collect()
    gc.freeze()
    if os.environ.get("NBFUZZ_LIST"):
        sys.__stderr__.write("FUNCS %s\nCLASSES %s\n" % (
            " ".join("%s.%s" % (f.__module__, f.__name__) for f in FUNCS),
            " ".join(c.__module__ + "." + c.__name__ for c, _ in CLASSES)))


def reset():
    # Drop every per-input object, then run the fixtures' own drains, so the next input starts from the
    # state the process had after startup. Left on purpose: counters nothing reads back or branches on
    # (keep_alive_cb_deleted) and UniqueInt::instances, whose expired weak_ptr entries act as absent.
    _inst.clear()
    gc.collect()
    for f in DRAIN_FUNCS:
        try:
            f()
        except Exception:
            pass
    if STATIC_PROPS is not None:
        try:
            STATIC_PROPS[0].value = STATIC_PROPS[1]
        except Exception:
            pass


def one(data):
    r = Reader(data)
    pool = []
    try:
        for _ in range(6):
            if not r.more():
                break
            try:
                op(r, pool)
            except Exception:
                pass
    finally:
        pool.clear()
        del pool
        reset()
    return 0
