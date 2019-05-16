import random

class RandomNumberGenerator(object):
    #Object wrappper around random module to simulate multiple
    #Random number generator objects with state
    def __init__(self, s=None):
        self.seed(s)

    def seed(self, s):
        random.seed(s)
        self.state=random.getstate()

    def randint(self,a,b):
        random.setstate(self.state)
        ret = random.randint(a,b)
        self.state=random.getstate()
        return ret

    def randnints(self,a,b,d):
        random.setstate(self.state)
        ret = [ random.randint(a,b) for _ in range(d)]
        self.state=random.getstate()
        return tuple(ret)
