def condition_to_flag(condition):
    """ Cast boolean to unsigned int.
    """
    return 1 if condition else 0

def get_dimensions(elements):
    """ Returns the number of elements in each dimension as a list, and implicitly also how many nested dimensions there are.
    Dimensions and ranks are in "little endian".
    The most inner / minor dimension (least significant) comes first and the most outer / major dimension (most significant) last.
    ----------
    elements : The nested lists.
    """
    dimensions = []
    inner_elements = elements
    while isinstance(inner_elements, list):
        dimensions.append(len(inner_elements))
        inner_elements = inner_elements[0]
    dimensions.reverse()
    return dimensions

def flatten_one_dimension(elements):
    return [lowerer_rank for middle_rank in elements for lowerer_rank in middle_rank]

def flatten_dimensions(elements, number_of_dimensions=None):
    """ Unrolls nested lists into one flat list.
    ----------
    elements : The nested lists.
    number_of_dimensions : How many of the outer dimensions to peel of, leaving the inner dimensions nested.
    """
    dimensions = get_dimensions(elements)
    if number_of_dimensions == None:
        number_of_dimensions = len(dimensions)
    for dimension_index in range(0, number_of_dimensions-1):
        elements = flatten_one_dimension(elements)
    return (dimensions[-number_of_dimensions:], elements)

def structure_one_dimension(dimension, elements):
    return [elements[offset:offset+dimension] for offset in range(0, len(elements), dimension)]

def structure_dimensions(dimensions, elements):
    """ Rolls a flat list back into nested lists.
    ----------
    dimensions : The number of elements in each dimension, as returned by get_dimensions(elements).
    elements : The flat list.
    """
    for dimension_index in range(0, len(dimensions)-1):
        elements = structure_one_dimension(dimensions[dimension_index], elements)
    return elements

def flatten_and_structure_dimensions(op, parameters, number_of_dimensions=None):
    """ Unrolls nested lists into one flat lists, applies the operation and rolls the resulting flat list back into nested lists.
    ----------
    op : Operation to apply to the tuple of flat lists, resulting in one flat list.
    parameters : The tuple flat lists.
    number_of_dimensions : How many of the outer dimensions to peel of, leaving the inner dimensions nested.
    """
    multi_args = isinstance(parameters, tuple)
    dimensions_and_arguments = map(lambda parameter: flatten_dimensions(parameter, number_of_dimensions), parameters if multi_args else [parameters])
    arguments = map(lambda (dimensions, argument): argument, dimensions_and_arguments)
    interleaved_arguments = zip(*arguments) if multi_args else arguments[0]
    processed_elements = op(interleaved_arguments)
    return structure_dimensions(dimensions_and_arguments[0][0], processed_elements)
