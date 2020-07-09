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

def inner_product(combination_op, reduction_op, accumulator, elements_a, elements_b):
    """ Computes the inner product of two nested lists of elements, by first combining them elementwise and then reducing everything into one element.
    ----------
    combination_op : Operation to apply to each element pair to get a combined element.
    reduction_op : Reduction operation to apply to each combined element and the accumulator. Must be associative and distributive.
    accumulator : Initial state of the reduction accumulator.
    elements_a : Nested lists of elements.
    elements_b : Nested lists of elements.
    """
    combined_elements = primitives.elementwise(combination_op, (elements_a, elements_b))
    (dimensions, flattened) = utils.flatten_dimensions(combined_elements)
    return primitives.reduction(reduction_op, flattened, accumulator)

def outer_product(combination_op, elements_a, elements_b, number_of_dimensions_a=None, number_of_dimensions_b=None):
    """ Computes the outer product of two nested lists of elements.
    ----------
    combination_op : Operation to apply to each element pair to get a combined element.
    elements_a : Nested lists of elements.
    elements_b : Nested lists of elements.
    number_of_dimensions_a : How many of the outer dimensions of elements_a to peel of, leaving the inner dimensions nested.
    number_of_dimensions_b : How many of the outer dimensions of elements_b to peel of, leaving the inner dimensions nested.
    """
    def for_each_element_a(combination_op, element_a, elements_b):
        return utils.flatten_and_structure_dimensions(lambda (flattened_b): primitives.elementwise(lambda (element_b): combination_op((element_a, element_b)), (flattened_b), 1), (elements_b), number_of_dimensions_b)
    return utils.flatten_and_structure_dimensions(lambda (flattened_a): primitives.elementwise(lambda (element_a): for_each_element_a(combination_op, element_a, elements_b), (flattened_a), 1), (elements_a), number_of_dimensions_a)

def dot_product(vector_a, vector_b):
    """ Computes the dot product of two vectors, resulting in a scalar.
    ----------
    vector_a : List of elements.
    vector_b : List of elements.
    """
    return inner_product(lambda (a, b): a*b, lambda (a, b): a+b, 0, vector_a, vector_b)

def matrix_multiplication(matrix_a, matrix_b):
    """ Computes the product of two matrices, resulting in another matrix.
    ----------
    matrix_a : Nested lists of elements.
    matrix_b : Nested lists of elements.
    """
    transposed_matrix_b = src_transposition([1, 0], matrix_b)
    return outer_product(lambda (vector_a, vector_b): dot_product(vector_a, vector_b), matrix_a, transposed_matrix_b, 1, 1)

def vector_transformation(matrix_a, vector_b):
    """ Computes the product of a matrix and a vector, resulting in another vector.
    ----------
    matrix_a : Nested lists of elements.
    vector_b : List of elements.
    """
    # Optimization: Transposition and inverse transposition cancel each other out
    # matrix_b = src_transposition([1, 0], [vector_b])
    # transformed_vector = matrix_multiplication(matrix_a, matrix_b)
    transformed_vector = outer_product(lambda (vector_a, vector_b): dot_product(vector_a, vector_b), matrix_a, [vector_b], 1, 1)
    # Optimization: Transpose and unpack the vector
    # return src_transposition([1, 0], transformed_vector)[0]
    return utils.flatten_dimensions(transformed_vector)[1]

def fft(elements, forward):
    """ Calculates the discrete fourier transform in O(n * log(n)).
    Note: The input size is restriced to powers of two.
    ----------
    elements : Complex element list.
    forward : If true the result will be scaled down by a factor of 1/len(elements).
    """
    radix = 2
    radix_stride = len(elements)/radix
    input_indices = range(0, radix_stride)
    gather_permutation_pairs = primitives.elementwise(lambda (input_index): [input_index, input_index+radix_stride], (input_indices), 1)
    gather_permutation = utils.flatten_dimensions(gather_permutation_pairs)[1]
    stage_size = 1
    while (stage_size < len(elements)):
        mask = stage_size-1
        indices_in_block = primitives.elementwise(lambda (index): index&mask, (input_indices), 1)
        block_base_indices = primitives.elementwise(lambda (input_index, index_in_block): input_index-index_in_block, (input_indices, indices_in_block), 1)
        offsets = primitives.elementwise(lambda (block_base_index): block_base_index*radix, (block_base_indices), 1)
        output_indices = primitives.elementwise(lambda (index_in_block, offset): index_in_block+offset, (indices_in_block, offsets), 1)
        scatter_permutation_pairs = primitives.elementwise(lambda (output_index): [output_index, output_index+stage_size], (output_indices), 1)
        scatter_permutation = utils.flatten_dimensions(scatter_permutation_pairs)[1]
        factor = 0.5/stage_size
        turns = primitives.elementwise(lambda (index_in_block): index_in_block*factor, (indices_in_block), 1)
        twiddle_factors = primitives.elementwise(lambda (turns): utils.twiddle_factor(turns), turns)
        dimensions = [radix, len(elements)/radix]
        permutated_elements = primitives.gather(gather_permutation, elements)
        permutated_pairs = utils.structure_dimensions(dimensions, permutated_elements)
        permutated_processed_pairs = primitives.elementwise(lambda ((element_a, element_b), twiddle_factor): utils.butterfly_op(element_a, element_b, twiddle_factor), (permutated_pairs, twiddle_factors), 1)
        permutated_processed_elements = utils.flatten_dimensions(permutated_processed_pairs)[1]
        processed_elements = primitives.scatter(scatter_permutation, permutated_processed_elements)
        elements = processed_elements
        stage_size *= radix
    if forward:
        factor = 1.0/len(elements)
        return primitives.elementwise(lambda (element): element*factor, (elements), 1)
    else:
        return elements

def convolution(elements_a, elements_b):
    """ Calculates the convolution of two signals in O(n * log(n)).
    Note: The input size is restriced to powers of two.
    ----------
    elements_a : Real or complex element list.
    elements_b : Real or complex element list.
    """
    a_is_complex = isinstance(elements_a[0], complex)
    b_is_complex = isinstance(elements_b[0], complex)
    factor = len(elements_a)
    transformed_elements_a = fft(elements_a, True)
    transformed_elements_b = fft(elements_b, True)
    transformed_result = primitives.elementwise(lambda (a, b): a*b*factor, (transformed_elements_a, transformed_elements_b), 1)
    result = fft(transformed_result, False)
    return result if a_is_complex or b_is_complex else map(lambda element: element.real, result)
