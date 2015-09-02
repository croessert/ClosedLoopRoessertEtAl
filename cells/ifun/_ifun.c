#include <Python.h>
#include <numpy/arrayobject.h>
#include "ifun.h"

/* Docstrings */
static char module_docstring[] = 
    "This module provides an interface for calculating the dynamic reservoir from Yamazaki and Tanaka 2005.";
static char ifun_docstring[] = 
    "Calculate the dynamic reservoir from Yamazaki and Tanaka 2005.";

/* Available functions */
static PyObject *ifun_ifun(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
    {"ifun", ifun_ifun, METH_VARARGS, ifun_docstring},
    {NULL, NULL, 0, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC init_ifun(void)
{
    PyObject *m = Py_InitModule3("_ifun", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}

static PyObject *ifun_ifun(PyObject *self, PyObject *args)
{
    int count, i;
    int N, T;    
    double Pr, tau, kappa;
    PyObject *I_obj, *r_obj, *ih_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OiidddOO", &r_obj, &N, &T, &Pr, &tau, &kappa, &ih_obj, &I_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *r_array = PyArray_FROM_OTF(r_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *I_array = PyArray_FROM_OTF(I_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *ih_array = PyArray_FROM_OTF(ih_obj, NPY_DOUBLE, NPY_IN_ARRAY);

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
    
    /* Get pointers to the data as C-types. */
    double *I    = (double*)PyArray_DATA(I_array);
    double *r    = (double*)PyArray_DATA(r_array);
    double *ih    = (double*)PyArray_DATA(ih_array);

    /* Call the external C function */
    double *z = ifun(r, N, T, Pr, tau, kappa, ih, I);

    /* Clean up. */
    Py_DECREF(I_array);
    Py_DECREF(r_array);
    Py_DECREF(ih_array);

    count = T*N;

    //FILE *file;
    //file = fopen("activity.dat", "w");
    //for(i = 0; i < count; i++){
    //  fprintf(file, "%.18f\n", z[i]);
    //}
    //fclose(file);

    PyObject *tuple = PyTuple_New(count);
    for (i = 0; i < count; i++) {
        double rVal = z[i];
        PyTuple_SetItem(tuple, i, Py_BuildValue("d", rVal));
    }

    //PyObject *result = PyTuple_New(count);
    //if (!result)
    //    return NULL;
    
    //for (i = 0; i < count; i++) {
    //    PyObject *value = PyFloat_FromDouble(z[i]);
    //    if (!value) {
    //        Py_DECREF(result);
    //        return NULL;
    //    }
    //    PyTuple_SetItem(result, i, value);
    //}

  //PyObject *result =  Py_BuildValue("d", 0.0);

  //return result; 
  return tuple; 

}