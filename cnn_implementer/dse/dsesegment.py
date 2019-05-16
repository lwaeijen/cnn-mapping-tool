from ..model import Segment

class DSESegment(Segment):
    def __init__(self, TESTING=False, **kwargs):

        #Init parent. Virtual segments contain nodes that are not in the real network (sources and sinks)
        super(DSESegment, self).__init__(TESTING=TESTING, **kwargs)

        # When testing is set, init a pseudo random number generator
        self.TESTING=TESTING
        if TESTING:
            import re
            from ..utils import RandomNumberGenerator
            seed=int(str(len(self))+''.join(map(lambda m: m.group(1), filter(lambda m: m!=None, [re.match(".*([0-9]+).*", n)  for n in self.nodes()]))))
            self.rng = RandomNumberGenerator(seed)

