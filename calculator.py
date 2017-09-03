import numpy
import numpy as np
import numpy.linalg
import scipy.sparse
import scipy as sp
from functools import reduce
import typing

def single_lottery(json):
    item_categories = create_categories([(3, 0.2), (4, 0.1)])
    return calculate_exact_expectation(item_categories)

Nary = typing.NamedTuple('Nary', (('number', int), ('cardinal', int)))

def is_valid(self):
    """
    >>> is_valid(Nary(1, 2))
    Nary(number=1, cardinal=2)
    >>> is_valid(Nary(-1, 2))
    Traceback (most recent call last):
     ...
    ValueError: Invalid Nary!: Nary(number=-1, cardinal=2)
    >>> is_valid(Nary(3, 2))
    Traceback (most recent call last):
     ...
    ValueError: Invalid Nary!: Nary(number=3, cardinal=2)
    """
    if -1 < self.number < self.cardinal:
        return self
    else:
        raise ValueError("Invalid Nary!: " + str(self))
Nary.is_valid = is_valid

def nary_to_int(nary):
    """
    >>> nary_to_int(list(map(lambda z: Nary(z[0], z[1]), [(1, 2), (0, 2), (1, 2)])))
    5
    >>> nary_to_int(list(map(lambda z: Nary(z[0], z[1]), [(3, 4), (0, 3), (1, 2)])))
    15
    """
    if nary == []:
        return 0
    head = nary[0].is_valid()
    tail = nary[1:]
    return head.number + head.cardinal * nary_to_int(tail)

def int_to_nary(num, cardinals):
    """
    >>> str(list(map(lambda z: (z.number, z.cardinal), int_to_nary(5, [2, 2 ,2]))))
    '[(1, 2), (0, 2), (1, 2)]'
    >>> str(list(map(lambda z: (z.number, z.cardinal), int_to_nary(15, [4, 3 ,2]))))
    '[(3, 4), (0, 3), (1, 2)]'
    >>> int_to_nary(8, [2, 2 ,2])
    Traceback (most recent call last):
     ...
    ValueError: num is too large to code!
    """    
    if cardinals == []:
        if num == 0:
            return []
        else:
            raise ValueError('num is too large to code!')
    head = cardinals[0]
    tail = cardinals[1:]
    return [Nary(num % head, head).is_valid()] + int_to_nary(num // head, tail)

def succ_nary(nary):
    """
    >>> succ_nary(Nary(2, 5))
    Nary(number=3, cardinal=5)
    >>> succ_nary(Nary(4, 5))
    Nary(number=4, cardinal=5)
    """
    return Nary(number=min(nary.number+1, nary.cardinal-1), cardinal=nary.cardinal)

ItemCategory = typing.NamedTuple('ItemCategory', (('number', int), ('probability', float)))

def to_prob(item_category):
    return item_category.number * item_category.probability

def total_of_item_category(item_categories):
    """
    >>> total_of_item_category(map(lambda z: ItemCategory(z[0], z[1]), [(3, 0.2), (4, 0.1)]))
    1.0
    """
    return sum(map(to_prob, item_categories))

def size_of_matrix(item_categories):
    """
    >>> size_of_matrix(list(map(lambda z: ItemCategory(z[0], z[1]), [(3, 0.2), (4, 0.1)])))
    20
    """
    if item_categories == []:
        return 1
    else:
        head = item_categories[0]
        tail = item_categories[1:]
        return (head.number + 1) * size_of_matrix(tail)
    
def prob_of_new_item(nary):
    """
    >>> prob_of_new_item(Nary(2, 3))
    0.0
    >>> prob_of_new_item(Nary(0, 3))
    1.0
    >>> prob_of_new_item(Nary(1, 5))
    0.75
    """
    nary.is_valid()
    return (nary.cardinal - nary.number - 1.0)/(nary.cardinal - 1.0)

def prob_of_duplication(nary):
    """
    >>> prob_of_duplication(Nary(2, 3))
    1.0
    >>> prob_of_duplication(Nary(0, 3))
    0.0
    >>> prob_of_duplication(Nary(1, 5))
    0.25
    """
    nary.is_valid()
    return 1.0 - prob_of_new_item(nary)

def iterator_over_state_space(cardinal_list):
    size = reduce(lambda u, v: u*v, cardinal_list)
    for i in range(size):
        yield int_to_nary(i, cardinal_list)
    
def item_categories_to_transition_matrix(item_categories):
    assert round(total_of_item_category(item_categories), 6) == 1.0
    
    size = size_of_matrix(item_categories)
    cardinal_list = list(map(lambda z: z.number + 1, item_categories))
    return_matrix = sp.sparse.lil_matrix((size, size))
    
    for state in iterator_over_state_space(cardinal_list):
        num_state = nary_to_int(state)
        for obtained_item in range(len(state)):
            probability_of_category = to_prob(item_categories[obtained_item])
            
            head = state[:obtained_item]
            body = state[obtained_item]
            foot = state[obtained_item+1:]
            new_item_state = head + [succ_nary(body)] + foot
            new_item_probability = probability_of_category * prob_of_new_item(body)
            return_matrix[nary_to_int(new_item_state), num_state] += new_item_probability
            
            duplicated_item_probability = probability_of_category * prob_of_duplication(body)
            return_matrix[num_state, num_state] += duplicated_item_probability
    
    return return_matrix.tocsr()

def create_q_matrix(transition_matrix):
    return (p[:-1, :-1], p[:-1, -1])

def I_Q_inv_e(q, solver):
    size = q.shape[0]
    return solver(scipy.sparse.eye(size) - q, np.ones([size]))

def exact_expectation(transition_matrix, solver=np.linalg.solve):
    q_matrix = transition_matrix[:-1, :-1]
    return I_Q_inv_e(q_matrix, solver)[0]

def create_categories(item_category_seed):
    item_categories = list(map(lambda z : ItemCategory(z[0], z[1]), item_category_seed))
    assert round(total_of_item_category(item_categories), 6) == 1.0
    return item_categories

def category_to_probability(item_category):
    return [item_category.probability] * item_category.number

def categories_to_probabilities(item_categories):
    return sum(map(category_to_probability, item_categories), [])

def calculate_exact_expectation(item_categories):
    import scipy.sparse.linalg
    transition_matrix = item_categories_to_transition_matrix(item_categories)
    return exact_expectation(transition_matrix.T, solver=scipy.sparse.linalg.spsolve)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
