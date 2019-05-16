import dill

#Class that can be saved to a file using dill
class Storable(object):

    @classmethod
    def load(self, fname):
        with open(fname, 'rb') as f:
            ret=dill.load(f)
        return ret

    def save(self, fname):
        #save to file using dill
        with open(fname, 'wb') as f:
            dill.dump(self, f)
