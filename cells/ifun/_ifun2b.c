#include <Python.h>
#include <numpy/arrayobject.h>
#include "ifun2b.h"

/* Docstrings */
static char module_docstring[] = 
    "This module provides an interface for calculating the dynamic reservoir from Yamazaki and Tanaka 2005.";
static char ifun2b_docstring[] = 
    "Calculate the dynamic reservoir from Yamazaki and Tanaka 2005.";

/* Available functions */
static PyObject *ifun2b_ifun2b(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
    {"ifun2b", ifun2b_ifun2b, METH_VARARGS, ifun2b_docstring},
    {NULL, NULL, 0, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC init_ifun2b(void)
{
    PyObject *m = Py_InitModule3("_ifun2b", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}

static PyObject *ifun2b_ifun2b(PyObject *self, PyObject *args)
{
    int count, i;
    int T;    
    PyObject *I_obj, *r_obj, *ih_obj, *C_obj, *tau_obj, *kappa_obj, *N_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOiOOOOO", &r_obj, &N_obj, &T, &C_obj, &tau_obj, &kappa_obj, &ih_obj, &I_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *I_array = PyArray_FROM_OTF(I_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *r_array = PyArray_FROM_OTF(r_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *ih_array = PyArray_FROM_OTF(ih_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *C_array = PyArray_FROM_OTF(C_obj, NPY_INT, NPY_IN_ARRAY);
    PyObject *tau_array = PyArray_FROM_OTF(tau_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *kappa_array = PyArray_FROM_OTF(kappa_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *N_array = PyArray_FROM_OTF(N_obj, NPY_INT, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (I_array == NULL) {
        Py_XDECREF(I_array);
        return NULL;
    }
    if (r_array == NULL) {
        Py_XDECREF(r_array);
        return NULL;
    }
    if (ih_array == NULL) {
        Py_XDECREF(ih_array);
        return NULL;
    }
    if (N_array == NULL) {
        Py_XDECREF(N_array);
        return NULL;
    }
    if (C_array == NULL) {
        Py_XDECREF(C_array);
        return NULL;
    }
    if (tau_array == NULL) {
        Py_XDECREF(tau_array);
        return NULL;
    }
    if (kappa_array == NULL) {
        Py_XDECREF(kappa_array);
        return NULL;
    }
    
    /* Get pointers to the data as C-types. */
    double *I    = (double*)PyArray_DATA(I_array);
    double *r    = (double*)PyArray_DATA(r_array);
    double *ih    = (double*)PyArray_DATA(ih_array);
    int *N    = (int*)PyArray_DATA(N_array);
    int *C    = (int*)PyArray_DATA(C_array);
    double *tau    = (double*)PyArray_DATA(tau_array);
    double *kappa    = (double*)PyArray_DATA(kappa_array);

    /* Call the external C function */
    double *z = ifun2b(r, N, T, C, tau, kappa, ih, I);

    /* Clean up. */
    Py_DECREF(I_array);
    Py_DECREF(r_array);
    Py_DECREF(ih_array);
    Py_DECREF(N_array);
    Py_DECREF(C_array);
    Py_DECREF(tau_array);
    Py_DECREF(kappa_array);

    count = T*(N[0]+N[1]);

    PyObject *tuple = PyTuple_New(count);

    for (i = 0; i < count; i++) {
      double rVal = z[i];
      PyTuple_SetItem(tuple, i, Py_BuildValue("d", rVal));
    }
    
    return tuple; 

}