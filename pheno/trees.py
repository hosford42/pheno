# =============================================================================
# Module:   pheno.trees
# File:     pheno/trees.py
# Author:   Aaron Hosford
# Created:  02/12/2015
#
# Copyright (c) Aaron Hosford 2015, all rights reserved.
# =============================================================================
#
# Description:
#   The pheno.trees submodule contains tools for applying evolutionary
#   algorithms to trees and forests.
#
# =============================================================================
#
# Modification History:
#
#   05/01/2012:
#     - Created this module from code originally appearing in __init__.py.
#
# =============================================================================

"""pheno.trees

The pheno.trees submodule contains tools for applying evolutionary algorithms
to trees and forests.
"""

# Future plans:
#   - Add a Forest class and use it instead of the dictionary of trees
#     currently used.
#   - Add a TreeEncoding, for when only one tree is desired as the phenotype.
#   - Revise the ForestEncoding class to make use of the TreeEncoding class.
#   - Move path generation out


# Standard library imports
import random

# Non-standard library imports

# Same project imports
from pheno import Address, Codon, Chromosome
from pheno import Genotype, CodonFactory, GeneticEncoding


__all__ = [
    'SpecialPathElement',
    'Path',
    'Tree',
    'ForestEncoding',
]


class SpecialPathElement:
    """A path element that is used to represent relative path elements and
    other, similarly non-standard behaviors for path elements."""

    def __init__(self, name):
        self._name = str(name)

    @property
    def name(self):
        """The name of the special path element."""
        return self._name

    def __str__(self):
        return self._name


class Path(Address):
    """Paths are addresses for trees."""

    current = SpecialPathElement('.')
    parent = SpecialPathElement('..')

    def __init__(self, elements=()):
        self._elements = tuple(elements)

    def __hash__(self):
        return hash(self._elements)

    def __eq__(self, other):
        # noinspection PyProtectedMember
        return isinstance(other, Path) and self._elements == other._elements

    def __ne__(self, other):
        return not isinstance(other, Path) or self._elements != other._elements

    def __str__(self):
        return self.to_str()

    @property
    def elements(self):
        """The path elements, in order, as a tuple."""
        return self._elements

    def to_str(self, separator='/'):
        """Return a string representation of the path, using the requested
        path element separator."""
        result = separator.join(str(element) for element in self._elements)
        if self._elements and self._elements[0] in (Path.current, Path.parent):
            return result
        else:
            return separator + result

    @property
    def is_root(self):
        """A boolean indicating whether the path points to the root of the tree."""
        return not self._elements

    def get_parent(self):
        """Return the parent of this path."""
        if not self._elements:
            return None
        return Path(self._elements[:-1])

    def get_child(self, element):
        """Construct and return a child path using the given element."""
        return Path(self._elements + (element,))

    def resolve(self, path):
        """Resolve the given relative path, using this path as the base."""
        if not isinstance(path, Path):
            raise TypeError(path)
        if path._elements and path._elements[0] in (Path.current, Path.parent):
            elements = list(self._elements)
        else:
            elements = []
        for element in path._elements:
            if element == Path.current:
                pass
            elif element == Path.parent:
                elements.pop()
            else:
                elements.append(element)
        return Path(elements)


class Tree:
    """A tree, as an evolutionary phenotype."""

    def __init__(self, value, subtrees=()):
        self._value = value
        self._subtrees = tuple(subtrees)

    @property
    def value(self):
        """The value stored at this node of the tree."""
        return self._value

    @property
    def subtrees(self):
        """The subtrees of this node of the tree, in order."""
        return self._subtrees

    def __str__(self):
        return self.to_str()

    def to_str(self, indentation=''):
        """A multi-line string representation of this tree, at the requested
        level of indentation."""
        result = indentation + str(self._value)
        for subtree in self._subtrees:
            result += '\n' + subtree.to_str(indentation + '  ')
        return result

    def size(self):
        """The size of this tree in nodes."""
        return 1 + sum([subtree.size() for subtree in self._subtrees])


class TreeCodonFactory(CodonFactory):
    """The codon factory used by ForestEncoding."""

    def __init__(self, value_generator, element_generator=None,
                 termination_condition=None, takes_address=False):
        self._value_generator = value_generator
        self._element_generator = element_generator
        self._termination_condition = termination_condition
        self._takes_address = bool(takes_address)

    @property
    def value_generator(self):
        """The function used to generate random values."""
        return self._value_generator

    @property
    def element_generator(self):
        """The function, if any, that is used to generate path elements."""
        return self._element_generator

    @property
    def termination_condition(self):
        """The function, if any, that is used to determine when a path should
        stop growing."""
        return self._termination_condition

    @property
    def takes_address(self):
        """A Boolean indicating whether the value generator expects the address
        as an argument."""
        return self._takes_address

    def get_random_path_element(self):
        """Get a random element for use in paths."""
        if self._element_generator:
            return self._element_generator()

        # Default element generation:
        if random.randrange(2):
            return random.random()
        elif random.randrange(2):
            return Path.current
        else:
            return Path.parent

    def get_random_address(self):
        """Get a random address."""
        elements = []
        while (self._termination_condition(elements)
               if self._termination_condition
               else random.randrange(2)):
            elements.append(self.get_random_path_element())
        return Path(elements)

    def get_random_value(self, address):
        """Get a random value."""
        if self._takes_address:
            return self._value_generator(address)
        else:
            return self._value_generator()


class ForestEncoding(GeneticEncoding):
    """
    Forest phenotype builder. Requires addresses to be paths.

    NOTE:
        This is an abstract class. You must override get_random_genotype() in
        the subclass.
    """

    def __init__(self, codon_factory, default_root=None):
        if not isinstance(codon_factory, TreeCodonFactory):
            raise TypeError(codon_factory)
        super().__init__(codon_factory)
        self.default_root = default_root

    def build_address_space(self, chromosome, default_root=None):
        """Build the address space for a chromosome."""
        address_space = {}
        current_path = Path()
        if default_root is None:
            default_root = self.default_root
        if default_root is not None:
            address_space[current_path] = default_root
        for codon in chromosome.codons:
            current_path = current_path.resolve(codon.address)
            address_space[current_path] = codon.value
        return address_space

    @staticmethod
    def prune_address_space(address_space, sorted_addresses=None):
        """Prune the address space."""
        if sorted_addresses is None:
            sorted_addresses = sorted(address_space, key=len)
        for address in sorted_addresses:
            if address.is_root or address.get_parent() in address_space:
                continue
            else:
                del address_space[address]

    @staticmethod
    def build_child_map(address_space):
        """Identify children of each address in the address space, placing them
        into a hierarchical map."""
        children = {}
        for address in address_space:
            if address.is_root:
                continue
            parent = address.get_parent()
            if parent in children:
                children[parent].add(address.elements[-1])
            else:
                children[parent] = {address.elements[-1]}
        return children

    @staticmethod
    def build_tree(children, address_space, sorted_addresses=None):
        """Build the tree from the child map."""
        if sorted_addresses is None:
            sorted_addresses = sorted(address_space, key=len)
        trees = {}
        for address in reversed(sorted_addresses):
            if address in children:
                subtrees = []
                for element in sorted(children[address]):
                    child = address.get_child(element)
                    subtrees.append(trees.pop(child))
                trees[address] = Tree(address_space[address], subtrees)
        return trees[Path()]

    def decode(self, genotype, default_root=None):
        """Construct the phenotype represented by the genotype."""
        forest = {}

        for chromosome_id in genotype.iter_chromosome_ids():
            chromosome = genotype.get_chromosome(chromosome_id)

            # Build address space
            address_space = self.build_address_space(chromosome, default_root)
            if Path() not in address_space:
                continue  # We can't build a tree without a value at the root

            # Sort the addresses
            sorted_addresses = sorted(address_space, key=len)

            # Prune address space
            self.prune_address_space(address_space, sorted_addresses)

            # Identify children
            children = self.build_child_map(address_space)

            # Build tree
            forest[chromosome_id] = self.build_tree(
                children,
                address_space,
                sorted_addresses
            )

        return forest

    def decompose_tree(self, tree, path=None, address_space=None):
        """Decompose a tree into a minimal address space that represents it."""
        if path is None:
            path = Path()
        if address_space is None:
            address_space = {}
        address_space[path] = tree.value
        elements = set()
        while len(elements) < len(tree.subtrees):
            elements.add(self.codon_factory.get_random_path_element())
        elements = sorted(elements)
        for subtree, element in zip(tree.subtrees, elements):
            subpath = path.get_child(element)
            self.decompose_tree(subtree, subpath, address_space)
        return address_space

    @staticmethod
    def determine_bloat(forest, target_size):
        """Determine how many non-coding codons to add to each forest's
        chromosome in order to reach the target size."""
        if not target_size:
            return {}
        total_size = sum(tree.size() for tree in forest.itervalues())
        total_bloat = target_size - total_size
        if total_bloat <= 0:
            return{}
        dividers = sorted(
            random.randrange(total_bloat + 1)
            for _ in range(len(forest))
        )
        bloat = {}
        for index, chromosome_id in enumerate(forest):
            if index:
                amount = dividers[index] - dividers[index - 1]
            else:
                amount = dividers[index]
            if amount:
                bloat[chromosome_id] = amount
        return bloat

    def apply_bloat(self, codons, address_space, amount):
        """Add random, non-coding codons."""
        while amount > 0:
            index = random.randint(0, len(codons))
            if index == len(codons) or random.randrange(2):
                path = random.choice(codons).address
                while (not path.elements or
                       path.get_parent() in address_space):
                    path.get_child(self.codon_factory.get_random_path_element())
                value = self.codon_factory.get_random_value(path)
                address_space[path] = value
                codon = Codon(path, value)
                codons.insert(index, codon)
            else:
                later_codon = random.choice(codons[index:])
                value = self.codon_factory.get_random_value(later_codon.address)
                codon = Codon(later_codon.address, value)
                codons.insert(index, codon)
            amount -= 1

    def encode(self, phenotype, target_size=None):
        """Construct a genotype that represents the phenotype."""
        if isinstance(phenotype, dict):
            forest = phenotype
        else:
            forest = dict(phenotype)

        bloat = self.determine_bloat(forest, target_size)

        chromosomes = {}
        for chromosome_id, tree in forest.items():
            address_space = self.decompose_tree(tree)

            addresses = address_space.keys()
            random.shuffle(addresses)

            codons = []
            for address in addresses:
                codon = Codon(address, address_space[address])
                codons.append(codon)

            if chromosome_id in bloat:
                self.apply_bloat(codons, address_space, bloat[chromosome_id])

            chromosomes[chromosome_id] = Chromosome(codons)

        return Genotype(chromosomes)
