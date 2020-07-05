import utils

def elementwise(op, parameters, number_of_dimensions=None):
    """ Repesents an operation which can be applied concurrently, because the invokations are independent and do not influence each other.
    Multiple parameter elements (represented as a tuple) can be combined into a single element in each invokation.
    ----------
    op : Operation to apply to each element tuple.
    parameters : Tuple of element lists, which will be transposed to element tuples.
    number_of_dimensions : How many of the outer dimensions to peel of, leaving the inner dimensions nested.
    """
    return utils.flatten_and_structure_dimensions(lambda arguments: map(op, arguments), parameters, number_of_dimensions)

def gather(permutation, elements):
    """ Permutation.
    Elements are discarded when no index in the permutation refers to them.
    Elements are replicated when multiple indices in the permutation refer to them.
    If len(permutation) < len(elements), then some elements must be discarded.
    If len(permutation) > len(elements), then some elements must be replicated.
    Elements can still be discarded and replicated even if len(permutation) == len(elements), depending on the indices.
    The output will have len(permutation) elements.
    ----------
    permutation : List of indices.
    elements : List of elements.
    """
    indices = range(0, len(permutation))
    return elementwise(lambda (index): elements[permutation[index]], (indices), 1)

def scatter(permutation, elements, condition_flags=None, result_size=None, default=0):
    """ Inverse of permutation.
    The permutation and elements must form matching pairs: len(permutation) == len(elements).
    If condition_flags are given, they must also match the permutation and elements: len(permutation) == len(result_size) and len(elements) == len(result_size).
    Elements are discarded when condition_flags are given and the specific flag of the element is set to false.
    Elements can not be replicated as each element can only be moved to one index (not a set of indices).
    If multiple indices collide, the resulting element at that index is undefined. Thus, this should be avoided.
    The output will have result_size elements or len(permutation) if result_size is not given.
    The resulting elements not referred to by any index are set to the default value.
    ----------
    permutation : List of indices.
    elements : List of elements.
    condition_flags : List of boolean flags.
    result_size : Expected number of flags set to "true" in condition_flags, for memory allocation.
    """
    if result_size == None:
        result_size = len(permutation)
    result = [default]*result_size
    def store(index):
        result[permutation[index]] = elements[index]
    def conditional_store(index):
        if condition_flags[index]:
            store(index)
    indices = range(0, len(permutation))
    elementwise(store if condition_flags == None else conditional_store, (indices), 1)
    return result

def reduction(op, elements, accumulator):
    """ Combines a list of elements into one using an accumulator. Returns the final state of the accumulator.
    ----------
    op : Operation to apply to each element and the accumulator. Must be associative and distributive.
    elements : List of elements.
    accumulator : Initial state of the accumulator.
    """
    for element in elements:
        accumulator = op((accumulator, element))
    return accumulator # reduce(op, elements, accumulator)

def integral(shift, op, elements, accumulator):
    """ Inverse of derivative.
    Scans a list of elements using an accumulator. Returns the integral of the elements and the final state of the accumulator.
    ----------
    shift : Shift of -1 is an exclusive scan and a shift of 1 is an inclusive scan.
    op : Operation to apply to each element and the accumulator. Must be associative and distributive.
    elements : List of elements.
    accumulator : Initial state of the accumulator.
    """
    integral = [0]*len(elements)
    if shift == -1:
        for i in range(0, len(elements)):
            integral[i] = accumulator
            accumulator = op((accumulator, elements[i]))
    elif shift == 1:
        for i in range(0, len(elements)):
            accumulator = op((accumulator, elements[i]))
            integral[i] = accumulator
    return (integral, accumulator)
