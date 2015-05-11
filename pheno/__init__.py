# =============================================================================
# Module:   pheno
# File:     pheno/__init__.py
# Author:   Aaron Hosford
# Created:  02/12/2015
#
# Copyright (c) Aaron Hosford 2015, all rights reserved.
# =============================================================================
#
# Description:
#   Evolutionary algorithm framework for Python. Enables construction of
#   arbitrarily complex, non-linear phenotypes from linear genotypes. As a
#   consequence of genotypic linearity, linear genetic operators from ordinary,
#   well understood genetic algorithms can be applied towards genetic
#   programming and other higher-dimensional constructions. Linearity of the
#   genotype is achieved by treating codons as "write" instructions addressing
#   into arbitrary locations in the structure/design space of the phenotype.
#
# =============================================================================
#
# Modification History:
#
#   05/01/2012:
#     - Created this module.
#   02/12/2015:
#     - Renamed the module from "genetic_design" to "pheno". Converted to
#       Python 3.3 and more appropriate coding standards. Refactored code,
#       moving some code out to submodules.
#
# =============================================================================

"""pheno

Pheno is an evolutionary algorithm framework for Python. Enables construction
of arbitrarily complex, non-linear phenotypes from linear genotypes. As a
consequence of genotypic linearity, linear genetic operators from ordinary,
well understood genetic algorithms can be applied towards genetic programming
and other higher-dimensional constructions. Linearity of the genotype is
achieved by treating codons as "write" instructions addressing into arbitrary
locations in the structure/design space of the phenotype.
"""


# Future plans:
#    - Permit composition of encodings, which allows us to successively map
#      from, for example, bit strings to path/value pairs to expression
#      forests, or any other sequence of encodings for which the individual
#      stages are implemented. This is important for several reasons, one of
#      which is that if we can start with a bit string representation, we can
#      use standard software/libraries designed for working with bit strings
#      on the genotypes.


# Standard library imports
import random
from abc import ABCMeta, abstractmethod

# Non-standard library imports

# Same project imports


__all__ = []


def select_proportionately(weight_map, total=None):
    """Select from among the keys of the weight map with probability
    proportional to the values. The weight map must be a dictionary with non-
    negative values. The total, if provided, should be the sum of the values in
    the weight map. (This is permitted for efficiency's sake.)"""
    if not isinstance(weight_map, dict):
        raise TypeError(weight_map)
    if not weight_map:
        raise ValueError(weight_map)

    if total is None:
        total = sum(weight_map.values())

    if total < 0:
        raise ValueError(weight_map)
    elif total:
        selector = random.uniform(0, total)
        for candidate, weight in weight_map.items():
            if selector <= weight:
                return candidate
            selector -= weight

    return random.choice(list(weight_map))


# A codon represents a single "write" instruction to the phenotype
class Codon:
    """Represents a single "write" instruction to the phenotype."""

    def __init__(self, address, value):
        self._address = address
        self._value = value

    @property
    def address(self):
        """Address to be written to."""
        return self._address

    @property
    def value(self):
        """Value to be written."""
        return self._value

    def __str__(self):
        return str(self._value) + "@" + str(self._address)


class Chromosome:
    """A sequence of codons, to be executed sequentially as a program."""

    def __init__(self, codons):
        self._codons = tuple(codons)

    @property
    def codons(self):
        """Codon sequence."""
        return self._codons

    def __str__(self):
        return self.to_str()

    def to_str(self, indentation=''):
        """Return a multi-line string representation at the requested
        indentation level."""
        if not self._codons:
            return ''
        return (
            indentation +
            ('\n' + indentation).join(str(codon) for codon in self._codons)
        )


class Genotype:
    """A sorted collection of chromosomes."""

    def __init__(self, chromosomes):
        self._chromosomes = dict(chromosomes)

    def __str__(self):
        return self.to_str()

    def to_str(self, indentation=''):
        """Return a multi-line string representation at the requested
        indentation level."""
        result = ''
        for chromosome_id in sorted(self._chromosomes):
            result += indentation + str(chromosome_id)
            chromosome = \
                self._chromosomes[chromosome_id].to_str(indentation + '  ')
            if chromosome:
                result += '\n' + chromosome
        return result

    def iter_chromosome_ids(self):
        """Return an iterator over the chromosome IDs."""
        return iter(self._chromosomes)

    def get_chromosome(self, chromosome_id):
        """Return the chromosome with the given ID, or None if no such
        chromosome exists."""
        return self._chromosomes.get(chromosome_id, None)


class CodonFactory(metaclass=ABCMeta):
    """Abstract base class for codon factories."""

    @abstractmethod
    def get_random_address(self):
        """Return a random address in the phenotype's address space."""
        raise NotImplementedError()

    @abstractmethod
    def get_random_value(self, address):
        """Return a random value that can be applied at the given address in
        the phenotype's address space."""
        raise NotImplementedError()

    def __call__(self):
        """Create a new codon randomly and return it, using values generated by
        get_random_address() and get_random_value()."""
        address = self.get_random_address()
        value = self.get_random_value(address)
        return Codon(address, value)


class GeneticEncoding(metaclass=ABCMeta):
    """Abstract base class for genetic factories for evolutionary algorithms."""

    def __init__(self, codon_factory):
        if not isinstance(codon_factory, CodonFactory):
            raise TypeError(codon_factory)
        self._codon_factory = codon_factory
        self._crossover_operators = {}
        self._mutation_operators = []

    @property
    def codon_factory(self):
        """The codon factory used to produce new codons when needed."""
        return self._codon_factory

    def add_crossover_operator(self, operator, weight=1):
        """Add a new crossover operator, used with probability proportional to
        its weight, to the exclusion of other crossover operators. Exactly one
        crossover operator is applied, never more, never less. If no crossover
        operator has been supplied, the default choice is to use proportional
        selection from among the unaltered parents."""
        self._crossover_operators[operator] = weight

    def add_mutation_operator(self, operator, probability=1):
        """Add a new mutation operator, applied with the given probability. All
        mutation operators are given a chance to be applied in the order they
        are added, regardless of whether previous mutation operators have
        already been applied."""
        self._mutation_operators.append((operator, probability))

    def get_children(self, parent_weight_map):
        """Create zero or more child genomes from those of the parents, with
        genetic material from each parent selected according to the given
        weights."""
        if not isinstance(parent_weight_map, dict):
            parent_weight_map = dict(parent_weight_map)

        if self._crossover_operators:
            operator = select_proportionately(self._crossover_operators)
            children = operator(parent_weight_map)
        else:
            # Default is no crossover.
            children = [select_proportionately(parent_weight_map)]

        mutated_children = []
        for child in children:
            for operator, probability in self._mutation_operators:
                if probability >= 1 or random.random() < probability:
                    child = operator(child, self.codon_factory)
            mutated_children.append(child)

        return mutated_children

    @abstractmethod
    def get_random_genotype(self):
        """Create a new, randomly generated genotype and return it."""
        # This is abstract to allow for variation in number of chromosomes,
        # length of chromosomes, and probability distribution thereof.
        raise NotImplementedError()

    @abstractmethod
    def decode(self, genotype):
        """Construct the phenotype represented by the genotype."""
        raise NotImplementedError()

    @abstractmethod
    def encode(self, phenotype):
        """Construct a genotype that represents the phenotype."""
        raise NotImplementedError()


class Address(metaclass=ABCMeta):
    """Abstract base class for addresses."""

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __ne__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError()


class CompositeAddress(Address):

    def __init__(self, components):
        self._components = tuple(components)
        for component in self._components:
            if not isinstance(component, Address):
                raise TypeError(component)

    def __hash__(self):
        return hash(self._components)

    def __eq__(self, other):
        # noinspection PyProtectedMember
        if (not isinstance(other, CompositeAddress) or
                len(self._components) != len(other._components)):
            return NotImplemented
        return self._components == other._components

    def __ne__(self, other):
        if (not isinstance(other, CompositeAddress) or
                len(self._components) != len(other._components)):
            return NotImplemented
        return self._components != other._components

    def __str__(self):
        return (
            '(' +
            ', '.join(str(component) for component in self._components) +
            ')'
        )


# These appear at the bottom because they rely on this module's contents
# already being defined.
import pheno.crossover
import pheno.mutation
import pheno.trees

