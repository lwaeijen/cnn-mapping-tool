import json

class Packing(object):
    #Helper that allows subclasses to be stored as json files

    def __init__(self, *args, **kwargs):
        if 'packed' in kwargs:
            self.unpack(kwargs['packed'])
        elif 'fname' in kwargs:
            self.load(kwargs['fname'])
        else:
            self.regular_init(*args, **kwargs)

    def __json_str(self):
        return json.dumps(self.pack(), indent=4, sort_keys=True)

    def __str__(self):
        return self.__json_str()

    def __repr__(self):
        return self.__str__()

    def save(self, fname):
        with open(fname, 'wt') as f:
            f.write(self.__json_str())

    def load(self, fname):
        with open(fname, 'rt') as f:
            packed=json.loads(f.read())
            self.unpack(packed)


    #################################
    # To be implemented by subclass
    #

    def pack(self):
        raise NotImplementedError("Subclass should implement this method")

    def unpack(self, packed):
        raise NotImplementedError("Subclass should implement this method")

    def regular_init(self, *args, **kwargs):
        raise NotImplementedError("Subclass should implement this method")

