# =============================================================================
# Module:   pheno.mutation
# File:     pheno/mutation.py
# Author:   Aaron Hosford
# Created:  02/12/2015
#
# Copyright (c) Aaron Hosford 2015, all rights reserved.
# =============================================================================
#
# Description:
#   The pheno.mutation submodule contains the mutation operators that come
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


'''pheno.mutation

The pheno.mutation submodule contains the mutation operators that come
packaged with pheno.
'''


# Future plans:


# Standard library imports
import random
from abc import ABCMeta, abstractmethod

# Non-standard library imports

# Same project imports
from pheno import Chromosome, Genotype


__all__ = [
    'MutationOperator',
    'UniformPointMutationOperator',
]

class MutationOperator(metaclass=ABCMeta): #pylint: disable=too-few-public-methods, abstract-class-little-used
    '''Abstract base class for all mutation operators.'''

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, original, codon_generator):
        raise NotImplementedError()


class UniformPointMutationOperator(MutationOperator): #pylint: disable=too-few-public-methods
    '''Uniform-probability point mutation operator.'''

    def __init__(self, point_probability=.001):
        super().__init__()
        self.point_probability = float(point_probability)

    def __call__(self, original, codon_generator):
        chromosomes = {}
        for chromosome_id in original.iter_chromosome_ids():
            chromosome = original.get_chromosome(chromosome_id)
            codons = []
            for codon in chromosome.codons:
                if random.random() < self.point_probability:
                    codon = codon_generator() #pylint: disable=not-callable
                codons.append(codon)
            chromosome = Chromosome(codons)
            chromosomes[chromosome_id] = chromosome
        return Genotype(chromosomes)

