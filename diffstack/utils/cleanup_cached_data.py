import sys
import dill
from dill._dill import StockUnpickler, _main_module, Pickler


def custom_load(file, ignore=None, replace_dict={}, **kwds):
    """
    Unpickle an object from a file.

    See :func:`loads` for keyword arguments.
    """
    return CustomUnpickler(file, ignore=ignore, replace_dict=replace_dict, **kwds).load()


class CustomUnpickler(StockUnpickler):
    """python's Unpickler extended to interpreter sessions and more types"""
    from dill.settings import settings
    _session = False

    def find_class(self, module, name):
        if (module, name) == ('__builtin__', '__main__'):
            return self._main.__dict__ #XXX: above set w/save_module_dict
        elif (module, name) == ('__builtin__', 'NoneType'):
            return type(None) #XXX: special case: NoneType missing
        if module == 'dill.dill': module = 'dill._dill'
        old_module = module        
        for name1, name2 in self._replace_dict.items():
            if module.startswith(name1):
                module = name2 + module[len(name1):]
        print (f"{old_module} --> {module} ({name})")
        return StockUnpickler.find_class(self, module, name)

    def __init__(self, *args, replace_dict={}, **kwds):
        settings = Pickler.settings
        _ignore = kwds.pop('ignore', None)
        StockUnpickler.__init__(self, *args, **kwds)
        self._main = _main_module
        self._ignore = settings['ignore'] if _ignore is None else _ignore
        self._replace_dict = replace_dict

    def load(self): #NOTE: if settings change, need to update attributes
        obj = StockUnpickler.load(self)
        if type(obj).__module__ == getattr(_main_module, '__name__', '__main__'):
            if not self._ignore:
                # point obj class to main
                try: obj.__class__ = getattr(self._main, type(obj).__name__)
                except (AttributeError,TypeError): pass # defined in a file
       #_main_module.__dict__.update(obj.__dict__) #XXX: should update globals ?
        return obj
    load.__doc__ = StockUnpickler.load.__doc__
    pass


if __name__ == '__main__':   

    # Needed to find model
    sys.path.append('./trajectron/trajectron')
    sys.path.append('./')

    args = sys.argv[1:]

    replace_dict = {
        "trajectron.trajectron": "diffstack.modules.predictors.trajectron_utils",
        "environment": "diffstack.modules.predictors.trajectron_utils.environment",
        "model": "diffstack.modules.predictors.trajectron_utils.model",
    }

    for filename in args:
        new_filename = filename + ".new"
        print (f"{filename} -> {new_filename}")

        # Import error can happen here is trying to load checkpoint with old cost_function object.
        # The old pred_metrics folder needs to be copied with environment/nuScenes_data/cost_functions.py     
        with open(filename, 'rb') as f:
            train_dataset = custom_load(f, replace_dict=replace_dict)

        print ("Loaded")

        with open(new_filename, 'wb') as f:
            dill.dump(train_dataset, f)

        print ("Saved")

    print ("done")
