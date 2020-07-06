import utils
import primitives

def derivative(shift, op, elements):
    """ Inverse of integral.
    Applies an operation to pairs of each element and an element shifted by a fixed amount of indices, wrapping at the edges.
    ----------
    shift : Number of indices to go from the current index.
    op : Operation to apply to each shifted and stationary element pair.
    elements : List of elements
    """
    if shift < 0:
        shift += len(elements)
    indices = range(0, len(elements))
    shifted_indices = primitives.elementwise(lambda (index): index+shift, (indices), 1)
    wrapped_indices = primitives.elementwise(lambda (index): index%len(elements), (shifted_indices), 1)
    shifted_elements = primitives.gather(wrapped_indices, elements)
    return primitives.elementwise(op, (shifted_elements, elements), 1)

def multi_dimensional(op, elements, number_of_dimensions=None):
    """ Allows to apply a 1D operation in n-D.
    ----------
    op : 1D operation to be applied to each dimension.
    elements : Nested lists of elements.
    number_of_dimensions : How many of the outer dimensions to peel of, leaving the inner dimensions nested.
    """
    if number_of_dimensions == None:
        number_of_dimensions = len(utils.get_dimensions(elements))
    for dimension in range(0, number_of_dimensions):
        processed_elements = primitives.elementwise(op, elements, number_of_dimensions-1)
        if dimension < number_of_dimensions-1:
            elements = src_transposition([dimension+1]+range(0, dimension+1)+range(dimension+2, number_of_dimensions), processed_elements)
        else:
            elements = src_transposition(range(number_of_dimensions-1, -1, -1), processed_elements)
    return elements

def element_index_to_ranks(dimensions, index):
    """ Converts a flat list index into a list of ranks for nested lists according to the given dimensions.
    ----------
    dimensions : The number of elements in each dimension, as returned by get_dimensions(elements).
    index : The index in the flat list.
    """
    (base_factors, max_index_plus_one) = primitives.integral(-1, lambda (a, b): a*b, dimensions, 1)
    quotients = primitives.elementwise(lambda (base_factor): index/base_factor, (base_factors), 1)
    return primitives.elementwise(lambda (quotient, dimension): quotient%dimension, (quotients, dimensions), 1)

def element_ranks_to_index(dimensions, ranks):
    """ Converts a list of nested lists ranks into a index for a flat list according to the given dimensions.
    ----------
    dimensions : The number of elements in each dimension, as returned by get_dimensions(elements).
    ranks : The list of nested lists ranks.
    """
    (base_factors, max_index_plus_one) = primitives.integral(-1, lambda (a, b): a*b, dimensions, 1)
    products = primitives.elementwise(lambda (base_factor, rank): base_factor*rank, (base_factors, ranks), 1)
    return primitives.reduction(lambda (a, b): a+b, products, 0)

def src_transposition(rank_permutation, elements):
    """ Gather for nested lists dimensions.
    Permutates the dimensions and ranks of nested lists.
    ----------
    rank_permutation : The indices of the src dimensions for each dst dimension.
    elements : The nested lists.
    """
    (src_dimensions, flattened) = utils.flatten_dimensions(elements)
    dst_dimensions = primitives.gather(rank_permutation, src_dimensions)
    number_of_elements = primitives.reduction(lambda (a, b): a*b, src_dimensions, 1)
    indices = range(0, number_of_elements)
    dst_ranks_per_element = primitives.elementwise(lambda (dst_index): element_index_to_ranks(dst_dimensions, dst_index), (indices), 1)
    src_ranks_per_element = primitives.elementwise(lambda (dst_ranks): primitives.scatter(rank_permutation, dst_ranks), (dst_ranks_per_element), 1)
    index_permutation = primitives.elementwise(lambda (src_ranks): element_ranks_to_index(src_dimensions, src_ranks), (src_ranks_per_element), 1)
    permutated_elements = primitives.gather(index_permutation, flattened)
    return utils.structure_dimensions(dst_dimensions, permutated_elements)

def dst_transposition(rank_permutation, elements):
    """ Scatter for nested lists dimensions.
    Permutates the dimensions and ranks of nested lists.
    ----------
    rank_permutation : The indices of the dst dimensions for each src dimension.
    elements : The nested lists.
    """
    (dst_dimensions, flattened) = utils.flatten_dimensions(elements)
    src_dimensions = primitives.scatter(rank_permutation, dst_dimensions)
    number_of_elements = primitives.reduction(lambda (a, b): a*b, dst_dimensions, 1)
    indices = range(0, number_of_elements)
    src_ranks_per_element = primitives.elementwise(lambda (src_index): element_index_to_ranks(src_dimensions, src_index), (indices), 1)
    dst_ranks_per_element = primitives.elementwise(lambda (src_ranks): primitives.gather(rank_permutation, src_ranks), (src_ranks_per_element), 1)
    index_permutation = primitives.elementwise(lambda (dst_ranks): element_ranks_to_index(dst_dimensions, dst_ranks), (dst_ranks_per_element), 1)
    permutated_elements = primitives.gather(index_permutation, flattened)
    return utils.structure_dimensions(src_dimensions, permutated_elements)
