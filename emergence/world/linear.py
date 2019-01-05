# N is the environment size
# K is the number of values each location can take
# r number of random non 0 locations
import random
import numpy as np

class Linear:

    @staticmethod
    def random_swap1():
        move = random.randint(0, 1) * 2 - 1
        def f(w):
            #random.choice([i for i, x in enumerate(w.world[:-1]) if x[0] == 0])
            try:
                if move > 0:
                    x = random.choice(
                            [i for i, x in enumerate(w.world[:-1]) if x[0] == 0])
                else:
                    x = random.choice(
                            [i for i, x in enumerate(w.world[1:]) if x[0] == 0])
                w.world[x], w.world[x+move] = (np.copy(w.world[x+move]),
                        np.copy(w.world[x]))
            except IndexError:
                pass
        return f

    @staticmethod
    def random_create():
        def f(w):
            non_objs = [i for i, x in enumerate(w.world) if x[0] == 1]
            not_present = set(range(w.depth)) - set(w.as_argmax())
            x = random.choice(non_objs)
            if w.unique_objs:
                w.world[x, random.choice(list(not_present))] = 1
            else:
                w.world[x, random.randint(1, w.depth-1)] = 1
            w.world[x, 0] = 0
        return f

    @staticmethod
    def random_destroy():
        def f(w):
            objs = [i for i, x in enumerate(w.world) if x[0] == 0]
            x = random.choice(objs)
            w.world[x, w.world[x].argmax()] = 0
            w.world[x, 0] = 1
        return f

    @staticmethod
    def swap(x, dist):
        def f(w):
            w.world[x], w.world[x+dist] = (np.copy(w.world[x+dist]),
                    np.copy(w.world[x]))
        return f

    @staticmethod
    def create(x, obj):
        def f(w):
            w.world[x, obj] = 1
            w.world[x, 0] = 0
        return f

    @staticmethod
    def destroy(x):
        def f(w):
            w.world[x, w.world[x].argmax()] = 0
            w.world[x, 0] = 1
        return f


    def __init__(self, size, depth, n_objects=None, unique_objs=True):
       self.size = size
       self.depth = depth
       self.unique_objs = unique_objs
       if n_objects is not None:
           if unique_objs:
               self.world = self.make_world(n_objects)
           else:
               self.world = self.make_random_world(n_objects)


    def make_world(self, n_objects):
        obj_locs = list(range(self.size))
        random.shuffle(obj_locs)
        obj_locs = obj_locs[:n_objects]
        objs = list(range(1, self.depth))
        random.shuffle(objs)

        world = np.zeros((self.size, self.depth))
        #world = np.zeros((self.size, n_objects+1))
        for i in range(self.size):
            if i in obj_locs:
                world[i, objs.pop()] = 1
            else:
                world[i, 0] = 1
        return world

    def make_random_world(self, n_objects):
        obj_locs = list(range(self.size))
        random.shuffle(obj_locs)
        obj_locs = obj_locs[:n_objects]
        world = np.zeros((self.size, self.depth))
        for i in range(self.size):
            if i in obj_locs:
                world[i, random.randint(1, self.depth-1)] = 1
            else:
                world[i, 0] = 1
        return world

    def as_argmax(self):
        return [np.argmax(x) for x in self.world]

    def __str__(self):
        return str(self.as_argmax())

    def __eq__(self, other):
        return (self.world == other.world).all()

    def copy(self):
        w = Linear(self.size, self.depth, None, self.unique_objs)
        w.world = np.copy(self.world)
        return w

    def apply(self, mutate):
        w = self.copy()
        mutate(w)
        return w

if __name__ == '__main__':
    w = Linear(10, 7, 4)
    print(w)
    print(w.apply(random_swap1()))
    print(w.apply(random_swap1()))
