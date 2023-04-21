import json
import os
from datetime import datetime

__all__ = ['params']


class params:
    """
    Central parameter class to distribute all core parameters
    anywhere in the project.

    Notes
    -----
    This is a prebuilt singleton object. Instanciating it will
    return the already existing instance. See examples below.
    Its content is being read and written from an internal json-file
    called `param_dict.json`. DO NOT MODIFY IT MANUALLY!

    Parameters get added to the objects as attributes during runtime.
    This means that you will be able to call a new attribute from
    `my_instance.add({'name': 'value'})` with `my_instance.name`.

    Furthermore, this is a purely local submodule and does not work
    outside of the avinoise module. Only use for other submodules.

    Examples
    --------
    >>> # print available parameters:
    >>> import avinoise.avinoise.config as avicfg
    >>> p = avicfg.params()
    >>> p.summary()

    >>> # add a single parameter to the list:
    >>> import avinoise.avinoise.config as avicfg
    >>> p = avicfg.params()
    >>> added_params = {"fs": 48000}
    >>> p.add(added_params)
    >>> p.summary()
    >>> # now access your parameter
    >>> print(p.fs)

    >>> # add multiple parameters to the list:
    >>> import avinoise.avinoise.config as avicfg
    >>> p = avicfg.params()
    >>> added_params = {"n_files": 10000,
                    "path": "/data"}
    >>> p.add(added_params)
    >>> p.summary()
    >>> # now access your parameters
    >>> print(p.n_files)
    >>> print(p.path)
    """

    # singleton guard
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(params, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.DICT_NAME = os.path.join("avinoise", "avinoise",
                                      "param_dict.json")
        # create default json if needed
        if not os.path.exists(self.DICT_NAME):
            last_modified = "modif"
            d = {last_modified: str(datetime.now())}
            setattr(self, last_modified, d[last_modified])
            self._save(d)
        else:
            self._updateattr()

    # get desired param as class variable
    def __getattr__(self, name):
        try:
            return self._open()[name]
        except KeyError:
            msg = "'{0}' object has no attribute '{1}'"
            raise AttributeError(msg.format(type(self).__name__, name))

    def _updateattr(self):
        d = self._open()
        for key in d:
            setattr(self, key, d[key])

    # dump to json file
    def _save(self, d):
        d["modif"] = str(datetime.now())
        with open(self.DICT_NAME, 'w+') as f:
            json.dump(d, f)

    # open json file as dict
    def _open(self):
        with open(self.DICT_NAME) as f:
            content = f.read()
        return json.loads(content)

    def _remove(self, param_name):
        d = self._open()
        d.pop(param_name)
        self._save(d)

    def _delete(self):
        d = self._open()
        d = {}
        self._save(d)

    def summary(self):
        '''
        Print out a summary of all parameters and
        their values.
        '''
        max_tabs = 3
        tab_len = 8
        print("")
        print("****************************Param summary****************************")
        print("\tName\t\t\t|\tValue")
        print("---------------------------------------------------------------------")
        d = self._open()
        for param in d:
            tabs = (max_tabs - ((len(param) // tab_len))) * "\t"
            print(f"\t{param}{tabs}|\t{d[param]}")
        print("*********************************************************************")
        print("")

    def add(self, param_dict):
        '''
        Add one or more parameters to the list

        Parameters
        ----------
        `param_dict`:
        Dictionary. Params in shape `{"name": value}`
        '''
        d = self._open() | param_dict
        self._save(d)
        self._updateattr()
