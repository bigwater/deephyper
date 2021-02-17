"""
Command line script to run Python function in an external process

Usage: python runner.py <modulePath> <moduleName> <funcName> <args>

Loads Python module <moduleName> located in the <modulePath> directory.
The function <funcName> must be a module-level attribute (e.g. not nested
inside a class), take one dictionary argument, and return a scalar objective
value. The passed dictionary is obtained by decoding <args>, which should be a
JSON-formatted dictionary escaped by single quotes.
"""
import importlib
import sys
import json

def load_module(name, path):
    try:
        mod = importlib.import_module(name)
    except (ModuleNotFoundError, ImportError):
        sys.path.insert(0, path)
        mod = importlib.import_module(name)
    return mod

if __name__ == "__main__":
    argv_cp = sys.argv[:]
    sys.argv = sys.argv[:1]
    modulePath = argv_cp[1]
    moduleName = argv_cp[2]

    print('module path = ', modulePath)
    print('module name = ', modulePath)

    module = load_module(moduleName, modulePath)

    funcName = argv_cp[3]
    print('func name = ', funcName)

    args = argv_cp[4]
    print('args = ', args)

    d = json.loads(args)
    func = getattr(module, funcName)
    print('func = ', func)

    retval = func(d)
    print("DH-OUTPUT:", retval)
