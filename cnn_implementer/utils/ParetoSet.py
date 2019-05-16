class ListNode(object):
    def __init__(self, data):
        self.next_node=None
        self.data=data

class LList(object):
    def __init__(self,*args):
        self.head=None
        for arg in args:
            self.append(arg)

    def append(self, data):
        n=ListNode(data)
        n.next_node=self.head
        self.head=n

    def __iter__(self):
        node=self.head
        while node:
            yield node.data
            node = node.next_node

    def __len__(self):
        cnt=0
        for _ in self:
            cnt+=1
        return cnt

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(list(self))


class ParetoSet(LList):
    def __init__(self,l=[]):
        super(ParetoSet, self).__init__()

        #dominate function can be overridden if desired
        self.dom=self.dominate
        self.cost_final_idx=None

        #initialize with given arguments
        for item in l:
            self.append(item)

    def dominate(self,a,b):
        for ai,bi in zip(a,b):
            if ai>bi:
                return False
        return True

    def append(self, new_cost):
        #loop over existing points and take action as required
        removed=False
        last=None
        node = self.head
        while node != None:
            old_cost = node.data
            if not removed and self.dom(old_cost, new_cost):
                #new cost is dominated by existing one, ignore new cost
                return
            if self.dom(new_cost, old_cost):
                #new cost dominates an existing point (or equal, in which case we don't care if we swap or not)
                #remove existing node and later insert the new cost
                if last:
                    #delete dominated point
                    del last.next_node #explicitly deleting to free memory
                    last.next_node=node.next_node
                else:
                    self.head=node.next_node
                removed=True
            else:
                last=node
            node=node.next_node

        #if we reach here the new cost should be inserted
        super(ParetoSet, self).append(new_cost)

    def __iadd__(self, other):
        #can either add complete other set
        if (type(other).__name__ == 'ParetoSet'):
            for _ in map(self.append, other):
                pass

        #Or the input is treated as a single element
        else:
            self.append(other)
        return self

class CNNParetoSet(ParetoSet):
    def __init__(self,*args):
        super(CNNParetoSet, self).__init__(*args)

    def dominate(self,a,b):
        for ai,bi in zip(a[:-1],b[:-1]):
            if ai>bi:
                return False
        return True

    def __iadd__(self, other):
        #can either add complete other set
        if (type(other).__name__ == 'CNNParetoSet'):
            for _ in map(self.append, other):
                pass

        #Or the input is treated as a single element
        else:
            self.append(other)
        return self


if __name__ == '__main__':
    from random import randint, seed
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    for i in range(3):
        print 'Running testcase %d out of 3'%(i)
        seed(1337+i)
        points=[ (randint(0,100000), randint(0,100000), None)  for _ in range(1000000)]

        #filter pareto
        print 'start filter'
        par=CNNParetoSet(points)
        print 'end filter'

        print 'plotting points'
        #helper to split points into two arrays
        def split(arr):
            x=[a for a,b in arr]
            y=[b for a,b in arr]
            return (x,y)

        def parsplit(arr):
            x=[a for a,b,_ in arr]
            y=[b for a,b,_ in arr]
            return (x,y)

        #clear the figure from any existing data/plots
        plt.clf()

        #plot regular points
        points_x, points_y=split(points)
        plt.scatter(points_x, points_y, zorder=1, c='gray')

        #plot pareto points
        par_x, par_y=parsplit(par)
        plt.scatter(par_x, par_y, zorder=2, c='red')

        #show the plot
        plt.show()
