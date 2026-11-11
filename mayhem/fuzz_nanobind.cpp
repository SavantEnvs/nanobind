// libFuzzer harness for nanobind: runs the bytes as a short program inside an embedded CPython that
// calls the extension modules nanobind's own test-suite builds (tests/test_*.cpp), so the fuzzed code is
// nanobind's argument casters, overload dispatch, ndarray/buffer import and container bindings.
// NBFUZZ_GROUP selects which test extensions mayhem/nbfuzz.py loads (functions | stl | ndarray).
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
#include <limits.h>

#ifndef NBFUZZ_GROUP
#define NBFUZZ_GROUP "functions"
#endif

// ---------------------------------------------------------------------------------------------
// nbfuzz_native: objects that expose arbitrary (but memory-consistent) buffer / DLPack metadata.
// ---------------------------------------------------------------------------------------------
namespace {

struct BufObject {
    PyObject_HEAD
    char *data;
    Py_ssize_t len;
    char *format;
    Py_ssize_t itemsize;
    int ndim;
    Py_ssize_t shape[4];
    Py_ssize_t strides[4];
    int strides_mode;
    int readonly;
};

int buf_getbuffer(PyObject *self, Py_buffer *view, int flags) {
    BufObject *o = (BufObject *) self;
    view->obj = self;
    Py_INCREF(self);
    view->buf = o->data;
    view->len = o->len;
    view->readonly = o->readonly;
    view->itemsize = o->itemsize;
    view->format = o->format;
    view->ndim = o->ndim;
    view->shape = o->ndim ? o->shape : nullptr;
    view->strides = (o->ndim && o->strides_mode) ? o->strides : nullptr;
    view->suboffsets = nullptr;
    view->internal = nullptr;
    if (!(flags & PyBUF_FORMAT) && !o->format)
        view->format = nullptr;
    return 0;
}

void buf_dealloc(PyObject *self) {
    BufObject *o = (BufObject *) self;
    PyTypeObject *tp = Py_TYPE(self);
    free(o->data);
    free(o->format);
    tp->tp_free(self);
    Py_DECREF(tp);
}

PyType_Slot buf_slots[] = {
    { Py_bf_getbuffer, (void *) buf_getbuffer },
    { Py_tp_dealloc, (void *) buf_dealloc },
    { 0, nullptr }
};
PyType_Spec buf_spec = { "nbfuzz_native.Buffer", sizeof(BufObject), 0, Py_TPFLAGS_DEFAULT, buf_slots };
PyTypeObject *buf_type;

void fill_strides(int ndim, const Py_ssize_t *shape, Py_ssize_t unit, int mode, Py_ssize_t *out) {
    // mode 1: C order, 2: Fortran order; `unit` is the stride unit (bytes for buffers, 1 for DLPack)
    if (mode == 2) {
        Py_ssize_t s = unit;
        for (int i = 0; i < ndim; ++i) { out[i] = s; s *= shape[i] ? shape[i] : 1; }
    } else {
        Py_ssize_t s = unit;
        for (int i = ndim - 1; i >= 0; --i) { out[i] = s; s *= shape[i] ? shape[i] : 1; }
    }
}

// buffer(data: bytes, format: bytes | None, itemsize: int, shape: tuple, strides_mode: int, readonly: int)
PyObject *native_buffer(PyObject *, PyObject *args) {
    Py_buffer data;
    PyObject *fmt, *shape;
    Py_ssize_t itemsize;
    int strides_mode, readonly;
    if (!PyArg_ParseTuple(args, "y*OnO!ii", &data, &fmt, &itemsize, &PyTuple_Type, &shape,
                          &strides_mode, &readonly))
        return nullptr;
    BufObject *o = PyObject_New(BufObject, buf_type);
    if (!o) { PyBuffer_Release(&data); return nullptr; }
    Py_INCREF(buf_type);
    o->format = nullptr;
    if (fmt != Py_None) {
        char *p; Py_ssize_t n;
        if (PyBytes_AsStringAndSize(fmt, &p, &n) == 0) {
            o->format = (char *) malloc((size_t) n + 1);
            memcpy(o->format, p, (size_t) n);
            o->format[n] = 0;
        } else {
            PyErr_Clear();
        }
    }
    if (itemsize < 1) itemsize = 1;
    o->itemsize = itemsize;
    o->ndim = (int) PyTuple_GET_SIZE(shape);
    if (o->ndim > 4) o->ndim = 4;
    Py_ssize_t count = 1;
    for (int i = 0; i < o->ndim; ++i) {
        o->shape[i] = PyLong_AsSsize_t(PyTuple_GET_ITEM(shape, i));
        if (o->shape[i] < 0) { PyErr_Clear(); o->shape[i] = 0; }
        count *= o->shape[i];
    }
    o->strides_mode = strides_mode;
    fill_strides(o->ndim, o->shape, itemsize, strides_mode, o->strides);
    o->len = count * itemsize;
    o->data = (char *) malloc((size_t) o->len + 1);
    memset(o->data, 0, (size_t) o->len + 1);
    memcpy(o->data, data.buf, (size_t) (data.len < o->len ? data.len : o->len));
    o->readonly = readonly;
    PyBuffer_Release(&data);
    return (PyObject *) o;
}

struct DLDevice { int32_t device_type; int32_t device_id; };
struct DLDataType { uint8_t code; uint8_t bits; uint16_t lanes; };
struct DLTensor {
    void *data; DLDevice device; int32_t ndim; DLDataType dtype;
    int64_t *shape; int64_t *strides; uint64_t byte_offset;
};
struct DLManaged {
    DLTensor dl_tensor; void *manager_ctx; void (*deleter)(DLManaged *);
};
struct DLManagedVersioned {
    struct { uint32_t major, minor; } version;
    void *manager_ctx; void (*deleter)(DLManagedVersioned *); uint64_t flags; DLTensor dl_tensor;
};

struct DLObject {
    PyObject_HEAD
    char *data;
    size_t size;
    int32_t devtype;
    DLDataType dtype;
    int ndim;
    int64_t shape[4];
    int strides_mode;
    uint64_t offset;
    int versioned;
};

void dl_dealloc(PyObject *self) {
    DLObject *o = (DLObject *) self;
    PyTypeObject *tp = Py_TYPE(self);
    free(o->data);
    tp->tp_free(self);
    Py_DECREF(tp);
}

// Each capsule owns its own copy of the tensor so that the consumer may outlive the DLObject.
struct DLBox {
    union { DLManaged m; DLManagedVersioned v; };
    int64_t dims[8];
    char *data;
};

void dl_delete(DLManaged *m) { DLBox *b = (DLBox *) m; free(b->data); free(b); }
void dl_delete_v(DLManagedVersioned *m) { DLBox *b = (DLBox *) m; free(b->data); free(b); }

void dl_capsule_destructor(PyObject *cap) {
    for (const char *name : { "dltensor", "dltensor_versioned" }) {
        if (PyCapsule_IsValid(cap, name)) {
            void *p = PyCapsule_GetPointer(cap, name);
            if (name[8]) ((DLManagedVersioned *) p)->deleter((DLManagedVersioned *) p);
            else ((DLManaged *) p)->deleter((DLManaged *) p);
        }
    }
}

PyObject *dl_dlpack(PyObject *self, PyObject *, PyObject *) {
    DLObject *o = (DLObject *) self;
    DLBox *b = (DLBox *) calloc(1, sizeof(DLBox));
    b->data = (char *) malloc(o->size + 1);
    memcpy(b->data, o->data, o->size + 1);
    DLTensor *t = o->versioned ? &b->v.dl_tensor : &b->m.dl_tensor;
    t->data = b->data;
    t->device = { o->devtype, 0 };
    t->ndim = o->ndim;
    t->dtype = o->dtype;
    t->shape = b->dims;
    for (int i = 0; i < o->ndim; ++i) b->dims[i] = o->shape[i];
    if (o->strides_mode) {
        Py_ssize_t tmp[4];
        Py_ssize_t sh[4];
        for (int i = 0; i < o->ndim; ++i) sh[i] = o->shape[i];
        fill_strides(o->ndim, sh, 1, o->strides_mode, tmp);
        t->strides = b->dims + 4;
        for (int i = 0; i < o->ndim; ++i) t->strides[i] = tmp[i];
    }
    t->byte_offset = o->offset;
    if (o->versioned) {
        b->v.version = { 1, 0 };
        b->v.deleter = dl_delete_v;
        return PyCapsule_New(&b->v, "dltensor_versioned", dl_capsule_destructor);
    }
    b->m.deleter = dl_delete;
    return PyCapsule_New(&b->m, "dltensor", dl_capsule_destructor);
}

PyObject *dl_device(PyObject *self, PyObject *) {
    return Py_BuildValue("(ii)", ((DLObject *) self)->devtype, 0);
}

PyMethodDef dl_methods[] = {
    { "__dlpack__", (PyCFunction) (void *) dl_dlpack, METH_VARARGS | METH_KEYWORDS, nullptr },
    { "__dlpack_device__", dl_device, METH_NOARGS, nullptr },
    { nullptr, nullptr, 0, nullptr }
};
PyType_Slot dl_slots[] = {
    { Py_tp_dealloc, (void *) dl_dealloc },
    { Py_tp_methods, (void *) dl_methods },
    { 0, nullptr }
};
PyType_Spec dl_spec = { "nbfuzz_native.DLPack", sizeof(DLObject), 0, Py_TPFLAGS_DEFAULT, dl_slots };
PyTypeObject *dl_type;

// dlpack(data, code, bits, lanes, shape, strides_mode, byte_offset, device_type, versioned)
PyObject *native_dlpack(PyObject *, PyObject *args) {
    Py_buffer data;
    int code, bits, lanes, strides_mode, devtype, versioned;
    unsigned long long offset;
    PyObject *shape;
    if (!PyArg_ParseTuple(args, "y*iiiO!iKii", &data, &code, &bits, &lanes, &PyTuple_Type, &shape,
                          &strides_mode, &offset, &devtype, &versioned))
        return nullptr;
    DLObject *o = PyObject_New(DLObject, dl_type);
    if (!o) { PyBuffer_Release(&data); return nullptr; }
    Py_INCREF(dl_type);
    o->dtype = { (uint8_t) code, (uint8_t) bits, (uint16_t) lanes };
    o->ndim = (int) PyTuple_GET_SIZE(shape);
    if (o->ndim > 4) o->ndim = 4;
    size_t count = 1;
    for (int i = 0; i < o->ndim; ++i) {
        long v = PyLong_AsLong(PyTuple_GET_ITEM(shape, i));
        if (v < 0) { PyErr_Clear(); v = 0; }
        o->shape[i] = v;
        count *= (size_t) v;
    }
    size_t itemsize = ((size_t) bits * (size_t) (lanes ? lanes : 1) + 7) / 8;
    o->size = (size_t) offset + count * itemsize;
    o->data = (char *) calloc(o->size + 1, 1);
    memcpy(o->data + offset, data.buf, data.len < (Py_ssize_t) (o->size - offset) ? (size_t) data.len : o->size - offset);
    o->strides_mode = strides_mode;
    o->offset = offset;
    o->devtype = devtype;
    o->versioned = versioned;
    PyBuffer_Release(&data);
    return (PyObject *) o;
}

PyMethodDef native_methods[] = {
    { "buffer", native_buffer, METH_VARARGS, nullptr },
    { "dlpack", native_dlpack, METH_VARARGS, nullptr },
    { nullptr, nullptr, 0, nullptr }
};
PyModuleDef native_def = { PyModuleDef_HEAD_INIT, "nbfuzz_native", nullptr, -1, native_methods,
                           nullptr, nullptr, nullptr, nullptr };

}  // namespace

extern "C" PyObject *PyInit_nbfuzz_native() {
    buf_type = (PyTypeObject *) PyType_FromSpec(&buf_spec);
    dl_type = (PyTypeObject *) PyType_FromSpec(&dl_spec);
    if (!buf_type || !dl_type) return nullptr;
    return PyModule_Create(&native_def);
}

static PyObject *g_one;

// The test fixtures keep C++ statics that depend on each other (tests/test_classes.cpp: `struct_tmp` is
// destroyed after `struct_destructed`, and ~Struct pushes to it while nb::is_alive() is still true because
// Python is never finalized here). Their destruction at libc exit() is not an input-dependent behavior, so
// leave through _exit() with the status exit() was given, before the statics run.
static void nbfuzz_fast_exit(int status, void *) {
    fflush(nullptr);
    _exit(status);
}

extern "C" int LLVMFuzzerInitialize(int *, char ***) {
    // Route Python's allocations through malloc so ASan sees them.
    setenv("PYTHONMALLOC", "malloc", 1);
    setenv("PYTHONHASHSEED", "0", 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);

    char exe[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    std::string dir = ".";
    if (n > 0) {
        exe[n] = 0;
        dir = exe;
        dir.resize(dir.rfind('/'));
    }

    PyImport_AppendInittab("nbfuzz_native", PyInit_nbfuzz_native);
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.install_signal_handlers = 0;
    PyStatus st = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(st)) {
        fprintf(stderr, "nbfuzz: Python initialization failed\n");
        abort();
    }

    std::string boot = "import sys\nsys.path[:0] = [" + std::string("'") + dir + "/build/fuzz/tests', '" +
                       dir + "/mayhem']\n";
    if (PyRun_SimpleString(boot.c_str()) != 0) abort();
    PyObject *mod = PyImport_ImportModule("nbfuzz");
    if (!mod) { PyErr_Print(); abort(); }
    PyObject *init = PyObject_GetAttrString(mod, "init");
    PyObject *res = init ? PyObject_CallFunction(init, "s", NBFUZZ_GROUP) : nullptr;
    if (!res) { PyErr_Print(); abort(); }
    Py_DECREF(res);
    Py_DECREF(init);
    g_one = PyObject_GetAttrString(mod, "one");
    if (!g_one) { PyErr_Print(); abort(); }
    // Registered after the extension modules loaded, so it runs before their statics are destroyed.
    on_exit(nbfuzz_fast_exit, nullptr);
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 4096) return 0;
    PyObject *res = PyObject_CallFunction(g_one, "y#", (const char *) data, (Py_ssize_t) size);
    if (!res) {
        PyErr_Clear();
        return 0;
    }
    Py_DECREF(res);
    return 0;
}
