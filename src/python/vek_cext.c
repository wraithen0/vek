/*
 * vek Python C extension — eliminates ctypes overhead.
 * Direct C function calls with numpy array buffer protocol.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stddef.h>

/* vek API from vek.h */
extern void  vek_init(void);
extern const char *vek_backend_name(void);
extern float vek_dot_f32(const float *a, const float *b, size_t n);
extern float vek_l2sq_f32(const float *a, const float *b, size_t n);
extern float vek_cosine_f32(const float *a, const float *b, size_t n);

static PyObject *py_dot_f32(PyObject *self, PyObject *args)
{
    PyArrayObject *a_arr, *b_arr;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a_arr, &PyArray_Type, &b_arr))
        return NULL;

    npy_intp n = PyArray_DIM(a_arr, 0);
    if (PyArray_DIM(b_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "arrays must have same length");
        return NULL;
    }

    const float *a = (const float *)PyArray_DATA(a_arr);
    const float *b = (const float *)PyArray_DATA(b_arr);
    return PyFloat_FromDouble((double)vek_dot_f32(a, b, (size_t)n));
}

static PyObject *py_l2sq_f32(PyObject *self, PyObject *args)
{
    PyArrayObject *a_arr, *b_arr;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a_arr, &PyArray_Type, &b_arr))
        return NULL;

    npy_intp n = PyArray_DIM(a_arr, 0);
    if (PyArray_DIM(b_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "arrays must have same length");
        return NULL;
    }

    const float *a = (const float *)PyArray_DATA(a_arr);
    const float *b = (const float *)PyArray_DATA(b_arr);
    return PyFloat_FromDouble((double)vek_l2sq_f32(a, b, (size_t)n));
}

static PyObject *py_cosine_f32(PyObject *self, PyObject *args)
{
    PyArrayObject *a_arr, *b_arr;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a_arr, &PyArray_Type, &b_arr))
        return NULL;

    npy_intp n = PyArray_DIM(a_arr, 0);
    if (PyArray_DIM(b_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "arrays must have same length");
        return NULL;
    }

    const float *a = (const float *)PyArray_DATA(a_arr);
    const float *b = (const float *)PyArray_DATA(b_arr);
    return PyFloat_FromDouble((double)vek_cosine_f32(a, b, (size_t)n));
}

static PyObject *py_init(PyObject *self, PyObject *args)
{
    vek_init();
    Py_RETURN_NONE;
}

static PyObject *py_backend(PyObject *self, PyObject *args)
{
    return PyUnicode_FromString(vek_backend_name());
}

static PyMethodDef methods[] = {
    {"init",       py_init,       METH_NOARGS,  "Initialize vek dispatch table"},
    {"backend",    py_backend,    METH_NOARGS,  "Return backend name string"},
    {"dot_f32",    py_dot_f32,    METH_VARARGS, "dot product of two f32 arrays"},
    {"l2sq_f32",   py_l2sq_f32,   METH_VARARGS, "squared L2 distance of two f32 arrays"},
    {"cosine_f32", py_cosine_f32, METH_VARARGS, "cosine similarity of two f32 arrays"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_vek_cext",
    "vek C extension — direct calls, no ctypes overhead", -1, methods
};

PyMODINIT_FUNC PyInit__vek_cext(void)
{
    import_array();
    return PyModule_Create(&module);
}
