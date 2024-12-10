"""
File: rabbits_foxes.py

Description: A biological model that animates life-cycles, reproduction, and deterioration of a population. This project
 can be used for future roles in determining what features are needed for population growth (or development in any sense).

Animation showing a field, populated with rabbits (blue circles) and foxes (red circles). As rabbits eat grass
and foxes hunt rabbits, the three species' population will fluctuate. User may parse the terminal to change starting
population and grass growth rate, to witness the impact of different features.

Once 1000 generations have existed, a bar chart of each population count will be made.
"""

import random as rnd
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

matplotlib.use('TkAgg')
WRAP = True  # When moving beyond the border, do we wrap around to the other size
OFFSPRING_R = rnd.randint(1, 2)  # The number of offspring when a rabbit reproduces
OFFSPRING_F = 1
SPEED = 1  # Number of generations per frame


class Animal:
    """Establish one animal class, distinguishing species by type"""

    def __init__(self, type, size, k):
        self.x = rnd.randrange(0, size)
        self.y = rnd.randrange(0, size)

        self.eaten = 0
        self.type = type
        self.cycles_since_eaten = 0
        self.starving = False
        self.size = size

        self.hunger_cycles = k

    def hunger(self):
        """Create a starving feature for foxes: if they havent eaten after k cycles, they are starving"""
        if self.type == 'fox':
            if self.cycles_since_eaten >= self.hunger_cycles:
                self.starving = True

    def reproduce(self):
        self.eaten = 0
        return copy.deepcopy(self)

    def eat(self, amount):
        self.eaten += amount
        if self.type == 'fox':
            self.cycles_since_eaten = 0

    def move(self):
        global move
        if self.type == 'rabbit':
            move = 1
        if self.type == 'fox':
            move = 2

        if WRAP:
            self.x = (self.x + rnd.choice([(-1 * move), 0, move])) % self.size
            self.y = (self.y + rnd.choice([(-1 * move), 0, move])) % self.size
        else:
            self.x = min(self.size - 1, max(0, (self.x + rnd.choice([(-1 * move), 0, move]))))
            self.y = min(self.size - 1, max(0, (self.y + rnd.choice([(-1 * move), 0, move]))))


class Field:
    """ A field is a patch of grass with 0 or more rabbits hopping around
    in search of grass """

    def __init__(self, size, grass_rate):
        self.rabbits = []
        self.foxes = []
        self.size = size

        self.field = np.ones(shape=(size, size), dtype=int)
        self.cycle_count = 0

        # Create an array to represent the field with foxes
        self.field_with_foxes = np.copy(self.field)
        self.field_with_rabbits = np.copy(self.field)

        self.grass_rate = grass_rate
        self.season = 0

    def add_rabbit(self, rabbit):
        self.rabbits.append(rabbit)

    def add_fox(self, fox):
        self.foxes.append(fox)

    def move(self):
        for r in self.rabbits:
            r.move()
        for f in self.foxes:
            f.move()

    def eat(self):
        """ All rabbits try to eat grass at their current location. Foxes ate any rabbits at current location """

        for r in self.rabbits:
            r.eat(self.field[r.x, r.y])
            self.field[r.x, r.y] = 0

        for f in self.foxes:
            for r in self.rabbits:
                if f.x == r.x and f.y == r.y:
                    self.rabbits.remove(r)
                    f.eat(1)
                    f.cycles_since_eaten = 0
                    break

    def survive(self):
        """ Rabbits that have not eaten die. Foxes who are starving die. """
        self.rabbits = [r for r in self.rabbits if r.eaten > 0]
        self.foxes = [f for f in self.foxes if not f.starving]

    def reproduce(self):
        born_r = []
        for r in self.rabbits:
            if r.eaten > 0:
                for _ in range(rnd.randint(1, OFFSPRING_R)):
                    born_r.append(r.reproduce())
        self.rabbits += born_r

        born_f = []
        for fox in self.foxes:
            if fox.eaten > 0:
                for _ in range(rnd.randint(1, OFFSPRING_F)):
                    born_f.append(fox.reproduce())
        self.foxes += born_f

    def season_change(self):
        """Cycle through seasons ( 0: spring, 1: summer, 2: fall, 3: winter)"""
        self.season = (self.season + 1) % 4

    def grow(self):
        """Adjust grass growth rate based on the current season"""
        if self.season in [0, 2]:  # Spring and autumn
            seasonal_growth_rate = self.grass_rate
        elif self.season == 1:  # Summer
            seasonal_growth_rate = self.grass_rate * 0.5  # Reduced growth in summer
        else:  # Winter
            seasonal_growth_rate = self.grass_rate * 0.2

        growloc = (np.random.rand(self.size, self.size) < seasonal_growth_rate) * 1
        self.field = np.maximum(self.field, growloc)

    def plot_populations(self):
        """Record and plot populations at a specific point in time"""

        if not self.foxes:
            fox_pop = 0
        else:
            fox_pop = len(self.foxes)

        if not self.rabbits:
            rabbit_pop = 0
        else:
            rabbit_pop = len(self.rabbits)

        grass_pop = np.sum(self.field)

        print("FOX: ", fox_pop, "Rabbit:", rabbit_pop, "Grass :", grass_pop)

        categories = ['Rabbits', 'Foxes', 'Grass']
        populations = [rabbit_pop, fox_pop, grass_pop]

        plt.figure()
        plt.bar(categories, populations)
        plt.title('Population Count of Ecosystem Species')
        plt.show()

    def generation(self):
        """ Run one generation of actions """

        # Clear the arrays representing the field with rabbits and foxes so that the empty space updates
        self.field_with_rabbits = np.copy(self.field)
        self.field_with_foxes = np.copy(self.field)

        # Update the positions of rabbits and foxes on the respective arrays
        for rabbit in self.rabbits:
            self.field_with_rabbits[rabbit.x, rabbit.y] = 2  # Assuming 2 represents rabbits
        for fox in self.foxes:
            self.field_with_foxes[fox.x, fox.y] = 3  # Assuming 3 represents foxes

        self.move()
        self.eat()
        # update a fox's eating cycles and check if they are starving
        for fox in self.foxes:
            fox.cycles_since_eaten += 1
            fox.hunger()
        self.survive()
        self.reproduce()
        self.grow()
        # add an additional count as a generation passes
        self.cycle_count += 1

        # change the season after 25 generations
        if self.cycle_count % 25 == 0:
            self.season_change()

        # plot the populations after 1000 generations
        if self.cycle_count == 1000:
            self.plot_populations()


def animate(i, field, im, speed, ax):
    for _ in range(speed):
        field.generation()

    rabbits = field.field_with_rabbits
    foxes = field.field_with_foxes
    generation_count = field.cycle_count

    total_field = np.maximum(np.maximum(field.field, rabbits), foxes)
    cmap = matplotlib.colors.ListedColormap(['tan', 'green', 'blue', 'red'])

    im.set_array(total_field)
    im.set_cmap(cmap)
    rabbits_count = len(field.rabbits)
    foxes_count = len(field.foxes)
    ax.set_title(
        "Generation: " + str(generation_count) + " Rabbits: " + str(rabbits_count) + " Foxes: " + str(foxes_count))
    return im,


def main():

    # create the parser and add arguments for different features
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", help="Size of the field", type=int)
    parser.add_argument("-k", "--k_value", help="Cycles a fox can survive with no food", type=int)
    parser.add_argument("-g", "--growth_rate", help="Grass growth rate", type=float)
    parser.add_argument("-f", "--foxes", help="initial number of foxes", type=int)
    parser.add_argument("-r", "--rabbits", help="initial number of rabbits", type=int)
    parser.add_argument("-sp", "--speed", help="How many generations should go by in each frame", type=int)

    # parse the command line
    args = parser.parse_args()

    # establish baselines for each variable
    field_size = 125
    if args.size:
        field_size = args.size
    k_value = 40
    if args.k_value:
        k_value = args.k_value
    growth_rate = 0.1
    if args.growth_rate:
        growth_rate = args.growth_rate
    foxes = 30
    if args.foxes:
        foxes = args.foxes
    rabbits = 5
    if args.rabbits:
        rabbits = args.rabbits
    speed = 1
    if args.speed:
        speed = args.speed

    print("Field size: " + str(field_size), "K value: " + str(k_value), "Growth: " + str(growth_rate),
          "Foxes: " + str(foxes), " Rabbits: " + str(rabbits), "Speed: " + str(speed))

    # Create the ecosystem
    field = Field(size=field_size, grass_rate=growth_rate)

    # Initialize with some rabbits
    for _ in range(rabbits):
        rabbit = Animal(type='rabbit', size=field_size, k=k_value)
        field.add_rabbit(rabbit)

    for _ in range(foxes):
        fox = Animal(type='fox', size=field_size, k=k_value)
        field.add_fox(fox)

    # Set up the image object
    array = np.ones(shape=(field_size, field_size), dtype=int)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = plt.imshow(array, cmap='PiYG', interpolation='hamming', aspect='auto', vmin=0, vmax=3)
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im, speed, ax), frames=10 ** 100, interval=1, repeat=True)
    plt.show()


if __name__ == '__main__':
    main()
