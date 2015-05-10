# =============================================================================
# Module:   pheno.crossover
# File:     pheno/crossover.py
# Author:   Aaron Hosford
# Created:  02/12/2015
#
# Copyright (c) Aaron Hosford 2015, all rights reserved.
# =============================================================================
#
# Description:
#   The pheno.crossover submodule contains the crossover operators that come
#   packaged with pheno.
#
# =============================================================================
#
# Modification History:
#
#   05/01/2012:
#     - Created this module from code originally appearing in __init__.py.
#
# =============================================================================
#!/usr/bin/env python


'''pheno.crossover

The pheno.crossover submodule contains the crossover operators that come
packaged with pheno.
'''


# Future plans:


# Standard library imports
import random
from abc import ABCMeta, abstractmethod

# Non-standard library imports

# Same project imports
from pheno import select_proportionately, Chromosome, Genotype


__all__ = [
    'CrossoverOperator',
    'NPointMessyCrossoverOperator',
]


class CrossoverOperator(metaclass=ABCMeta): #pylint: disable=too-few-public-methods, no-init, abstract-class-little-used
    '''Abstract base class for crossover operators.'''

    @abstractmethod
    def __call__(self, parent_weight_map):
        raise NotImplementedError()


class NPointMessyCrossoverOperator(CrossoverOperator):
    '''N-point variable-length crossover.'''

    def __init__(self, points=2):
        self._points = int(points)

    @property
    def points(self):
        '''The number of crossover points.'''
        return self._points

    @staticmethod
    def get_total_weight(parent_weight_map, include=None):
        '''Total up the weight for the included parents.'''
        if include is None:
            include = parent_weight_map
        return float(sum(parent_weight_map[parent] for parent in include))

    def select_parents(self, candidates, parent_weight_map, total_weight=None):
        '''Select which parents are to contribute to the child's chromosome.'''
        if total_weight is None:
            total_weight = self.get_total_weight(parent_weight_map, candidates)

        # Pick which parents to use
        selected_parents = []
        while len(selected_parents) <= self._points:
            selected_parents.append(
                select_proportionately(parent_weight_map, total_weight)
            )

        return selected_parents

    def determine_crossover_points(self, selected_parents, chromosome_id):
        '''Select crossover points for each parent's chromosome.'''
        points = {}
        for parent in selected_parents:
            points[parent] = sorted(
                random.randint(0, len(parent.get_chromosome(chromosome_id)))
                for index in range(self._points)
            )
        return points

    def perform_crossover(self, selected_parents, points, chromosome_id):
        '''Perform the crossover operation to create a new chromosome.'''
        # Perform crossover on chromosomes
        child_codons = ()
        start = 0
        for index in range(self._points):
            parent = selected_parents[index]
            end = points[parent][index]
            child_codons += \
                parent.get_chromosome(chromosome_id).codons[start:end]
            start = end
        parent = selected_parents[-1]
        child_codons += parent.get_chromosome(chromosome_id).codons[end:]
        return Chromosome(child_codons)

    def new_chromosome(self, chromosome_id, parent_weight_map, total_weight=None): #pylint: disable=line-too-long
        '''Probabilistically create a new chromosome from the parents' based on
        their respective weights.'''
        if total_weight is None:
            total_weight = self.get_total_weight(parent_weight_map)

        # Get a list of parents with that chromosome
        parents_with_chromosome = [
            parent
            for parent in parent_weight_map
            if parent.has_chromosome_id(chromosome_id)
        ]

        # Determine whether to skip the given chromosome
        total_included_weight = self.get_total_weight(
            parent_weight_map,
            parents_with_chromosome
        )
        if random.uniform(0.0, total_weight) >= total_included_weight:
            return None

        # Pick which parents to use
        selected_parents = self.select_parents(
            parents_with_chromosome,
            parent_weight_map,
            total_included_weight
        )

        # Pick crossover points for each parent's chromosome
        points = self.determine_crossover_points(
            selected_parents,
            chromosome_id
        )

        # Perform crossover on chromosome
        return self.perform_crossover(
            selected_parents,
            points,
            chromosome_id
        )

    def __call__(self, parent_weight_map):
        '''Apply the crossover operator to the parents, biasing selection of
        genetic material from each parent according to its given weight.

        NOTE:
            Fitnesses must be >= 0.0. For the unweighted case, use 1.0 for each
            parent, not 0.0.
        '''
        if not isinstance(parent_weight_map, dict):
            parent_weight_map = dict(parent_weight_map)

        # Pre-compute total weight for efficiency
        total_weight = self.get_total_weight(parent_weight_map)

        # Determine all possible chromosome IDs
        chromosome_ids = set()
        for parent in parent_weight_map:
            chromosome_ids |= set(parent.iter_chromosom_ids())

        # For each chromosome ID
        child_chromosomes = {}
        for chromosome_id in chromosome_ids:
            # Add a new child chromosome
            chromosome = self.new_chromosome(
                chromosome_id,
                parent_weight_map,
                total_weight
            )
            if chromosome is not None:
                child_chromosomes[chromosome_id] = chromosome

        # We are expected to return a sequence of children, to permit the
        # implementation of non-lossy genetic crossover operators, i.e. those
        # that preserve all genetic information of the parents, and also to
        # permit the return of empty lists in the case of child validation
        # checking.
        return (Genotype(child_chromosomes),)
