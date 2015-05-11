# =============================================================================
# Module:   pheno.treegp
# File:     pheno/treegp.py
# Author:   Aaron Hosford
# Created:  02/12/2015
#
# Copyright (c) Aaron Hosford 2015, all rights reserved.
# =============================================================================
#
# Description:
#   The pheno.treegp submodule contains tools for applying evolutionary
#   algorithms to program trees and forests.
#
# =============================================================================
#
# Modification History:
#
#   05/01/2012:
#     - Created this module.
#
# =============================================================================

"""pheno.treegp

The pheno.treegp submodule contains tools for applying evolutionary algorithms
to program trees and forests.
"""


# Future plans:
#    - ArgumentConstraints still doesn't support more general constraints,
#      such as that different arguments possess the same unspecified type,
#      etc. However, it is good enough for current usage.
#    - Add symbols. Their values should depend on the execution context passed
#      in during evaluation. Some code was previously written for this, which
#      I have archived for safe keeping, but I didn't feel the design was
#      sufficiently well thought out to include at this point, so it has been
#      removed from this branch. For now, it should suffice to either use
#      hand-tailored functions, one per symbol, which return the symbols'
#      values, or use some sort of indexing into a value list.


# Standard library imports
import random

# Non-standard library imports

# Same project imports
from pheno import select_proportionately
from pheno.trees import Tree, ForestEncoding


__all__ = [
]


class Function:
    """A Python function, suitably wrapped for use in genetic programming."""

    def __init__(self, name, implementation, arg_constraints, return_type=None,
                 control=False):
        self.name = str(name)
        self.implementation = implementation
        self.arg_constraints = arg_constraints
        self.return_type = return_type or Type.anything()
        self.control = bool(control)
        if not isinstance(return_type, Type) or callable(return_type):
            raise TypeError(return_type)

    def __str__(self):
        return self.name

    def get_return_type(self, arg_types):
        """Determine the return type given the argument types, and return it."""
        # Assume arg_types is legal.
        if callable(self.return_type):
            return self.return_type(arg_types)
        else:
            return self.return_type

    def __call__(self, arg_expressions, context):
        if self.control:
            args = arg_expressions
        else:
            args = [arg_exp.evaluate(context) for arg_exp in arg_expressions]
        return self.implementation(args, context)


class Literal:

    def __init__(self, value, value_type=None):
        self.value = value
        self.value_type = value_type or Type.anything()
        if not isinstance(value_type, Type):
            raise TypeError(value_type)


class LiteralFactory:

    def __init__(self, generator, value_type=None):
        self.generator = generator
        self.value_type = value_type or Type.anything()
        if not callable(self.generator):
            raise TypeError(self.generator)
        if not isinstance(self.value_type, Type):
            raise TypeError(self.value_type)

    def get_random_literal(self):
        return Literal(self.generator(), self.value_type)


class ProgramNodeFactory:
    """Program node factory. Used to produce randomly selected functions and
    literals from which a program may be constructed."""

    def __init__(self):
        self._functions = {}
        self._literal_factories = {}
        self._combined_registry = {}
        self._total_function_weight = 0
        self._total_literal_factory_weight = 0
        self._total_weight = 0

    def register_function(self, function, weight=1):
        if not isinstance(function, Function):
            raise TypeError(function)
        weight = float(weight)
        self._functions[function] = weight
        self._combined_registry[function] = weight
        self._total_function_weight += weight
        self._total_weight += weight

    def register_literal_factory(self, literal_factory, weight=1):
        if not isinstance(literal_factory, LiteralFactory):
            raise TypeError(literal_factory)
        weight = float(weight)
        self._literal_factories[literal_factory] = weight
        self._combined_registry[literal_factory] = weight
        self._total_literal_factory_weight += weight
        self._total_weight += weight

    def get_random_value(self):
        item = select_proportionately(
            self._combined_registry,
            self._total_weight
        )
        if isinstance(item, Function):
            return item
        else:
            return item.get_random_literal()

    def get_random_function(self):
        return select_proportionately(
            self._functions,
            self._total_function_weight
        )

    def get_random_literal(self):
        factory = select_proportionately(
            self._literal_factories,
            self._total_literal_factory_weight
        )
        return factory.get_random_literal()


# TODO: Gut this, and then add type agreement enforcement. If compilation flag
#       is set, return a compilable tree, otherwise, return an interpretable
#       tree.
class ProgramForestEncoding(ForestEncoding):
    """
    Program forest phenotype builder. Addresses are paths, and values are
    functions or literal values.
    """

    def __init__(self, default_root=None):
        super().__init__()
        self.default_root = default_root

    @staticmethod
    def get_random_path_element():
        """Get a random element for use in paths."""
        if random.randrange(2):
            return random.random()
        elif random.randrange(2):
            return Path.current
        else:
            return Path.parent

    def get_random_address(self):
        """Get a random address."""
        elements = []
        while random.randrange(2):
            elements.append(self.get_random_path_element())
        return Path(elements)

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
            elements.add(self.get_random_path_element())
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
                    path.get_child(self.get_random_path_element())
                value = self.get_random_value(path)
                address_space[path] = value
                codon = Codon(path, value)
                codons.insert(index, codon)
            else:
                later_codon = random.choice(codons[index:])
                value = self.get_random_value(later_codon.address)
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
