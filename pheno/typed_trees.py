# =============================================================================
# Module:   pheno.typed_trees
# File:     pheno/typed_trees.py
# Author:   Aaron Hosford
# Created:  02/12/2015
#
# Copyright (c) Aaron Hosford 2015, all rights reserved.
# =============================================================================
#
# Description:
#   The pheno.typed_trees submodule contains tools for applying evolutionary
#   algorithms to typed trees and forests.
#
# =============================================================================
#
# Modification History:
#
#   02/24/2015:
#     - Created this module as a copy of pheno/trees.py. Modified code to
#       incorporate type constraints.
#
# =============================================================================

"""pheno.trees

The pheno.trees submodule contains tools for applying evolutionary algorithms
to trees and forests.
"""

# Future plans:
#    - Add a Forest class and use it instead of the dictionary of trees
#      currently used.
#    - Add a TreeEncoding, for when only one tree is desired as the phenotype.
#    - Revise the ForestEncoding class to make use of the TreeEncoding class.
#    - Move path generation out


# Standard library imports
import random

# Non-standard library imports

# Same project imports
from pheno import Address, Codon, Chromosome, CompositeAddress
from pheno import Genotype, CodonFactory, GeneticEncoding
from pheno.trees import SpecialPathElement, Path, Tree


__all__ = [
]


class Type:

    _anything = None

    @classmethod
    def anything(cls):
        if cls._anything is None:
            cls._anything = Type("anything", "The all-inclusive root type.")
            cls._anything._parents = frozenset()
        return cls._anything

    def __init__(self, name, description=None, parents=None, children=None):
        self._name = name
        self._description = description
        self._parents = {self.anything()}
        if parents:
            for parent in parents:
                self.inherit_from(parent)
        if children:
            for child in children:
                if not isinstance(child, Type):
                    raise TypeError(child)
                child.inherit_from(self)

    @property
    def name(self):
        """The name of this type."""
        return self._name

    @property
    def description(self):
        """A short description of this type."""
        return self._description

    def __str__(self):
        return self._name

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return self is other

    def __ne__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return self is not other

    def __le__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return self is other or self.has_descendant(other)

    def __ge__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return self is other or self.has_ancestor(other)

    def __lt__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return self.has_descendant(other)

    def __gt__(self, other):
        if not isinstance(other, Type):
            return NotImplemented
        return self.has_ancestor(other)

    def inherit_from(self, other):
        """Add the other type as a direct ancestor of this one."""
        if not isinstance(other, Type):
            raise TypeError(other)
        if self <= other or self is self.anything():
            raise ValueError(other)
        self._parents.add(other)

    def has_ancestor(self, other):
        """Return whether there is any path of inheritance from the other type
        to this one."""
        if not isinstance(other, Type):
            raise TypeError(other)

        # This case happens often enough that we should just go ahead and check
        # for it specifically.
        if other is self.anything():
            return other is not self

        queue = set(self._parents)
        covered = set()
        while queue:
            ancestor = queue.pop()
            if ancestor is other:
                return True
            if ancestor not in covered:
                queue |= ancestor._parents
                covered.add(ancestor)
        return False

    def has_descendant(self, other):
        """Return whether there is any path of inheritance from this type to
        the other one."""
        if not isinstance(other, Type):
            raise TypeError(other)
        return other.has_ancestor(self)


class ChildConstraints:
    """Represents a set of constraints that apply to the children of a tree node."""

    def __init__(self, min=0, max=None, types=None, repeat_type=None):
        self._min = int(min or 0)
        self._max = None if max is None else int(max)
        if self._max is not None and self._min > self._max:
            raise ValueError("Minimum arg count must be less than maximum.")
        if self._min < 0:
            raise ValueError("Minimum arg count must be non-negative.")

        if types is None:
            types = ()
        else:
            types = tuple(
                type or Type.anything() for type in types
            )
            for type in types:
                if not isinstance(type, Type):
                    raise TypeError(type)
        self._types = types

        self._repeat_type = repeat_type or Type.anything()
        if not isinstance(self._repeat_type, Type):
            raise TypeError(repeat_type)

    @property
    def min(self):
        """The minimum number of children permitted."""
        return self._min

    @property
    def max(self):
        """The maximum number of children permitted, or None if unlimited."""
        return self._max

    @property
    def types(self):
        """The initial child types. If a child index falls within the length of
        this tuple, its type is governed by the type at the corresponding index
        of this tuple."""
        return self._types

    @property
    def repeat_type(self):
        """The repeated child type. If an child index is greater than the
        length of the types tuple, its type is governed by this type."""
        return self._repeat_type

    def satisfied_by(self, types):
        """Return whether the given sequence of child types satisfies the child
        constraints."""
        if not isinstance(types, (list, tuple)):
            types = tuple(types)
        if len(types) < self._min:
            return False
        if self._max is not None and len(types) > self._max:
            return False
        for index, type in enumerate(types):
            if index < len(self._types):
                required_type = self._types[index]
            else:
                required_type = self._repeat_type
            if not required_type <= type:
                return False
        return True


class TypedTree(Tree):
    """A typed tree, as an evolutionary phenotype."""

    def __init__(self, type, value, subtrees=()):
        super().__init__(value, subtrees)
        self._type = type

    @property
    def type(self):
        """The type of this node of the tree."""
        return self._type

    def to_str(self, indentation=''):
        """A multi-line string representation of this tree, at the requested
        level of indentation."""
        result = indentation + str(self._value)
        if self._type.name[:1].lower() in 'aeiou':
            result += ' as an '
        else:
            result += ' as a '
        result += self._type.name
        for subtree in self._subtrees:
            result += '\n' + subtree.to_str(indentation + '  ')
        return result


class TypedTreeCodonFactory(CodonFactory):
    """The codon factory used by ForestEncoding."""

    def __init__(self, value_generator, element_generator=None,
                 type_generator=None, termination_condition=None):
        self._value_generator = value_generator
        self._element_generator = element_generator
        self._type_generator = type_generator
        self._termination_condition = termination_condition

    @property
    def value_generator(self):
        """The function used to generate random values."""
        return self._value_generator

    @property
    def element_generator(self):
        """The function, if any, that is used to generate path elements."""
        return self._element_generator

    @property
    def type_generator(self):
        """The function, if any, that is used to generate types."""
        return self._type_generator

    @property
    def termination_condition(self):
        """The function, if any, that is used to determine when a path should
        stop growing."""
        return self._termination_condition

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
        if self._type_generator:
            return CompositeAddress((Path(elements), self._type_generator()))
        else:
            return Path(elements)

    def get_random_value(self, address):
        """Get a random value."""
        return self._value_generator(address)


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
