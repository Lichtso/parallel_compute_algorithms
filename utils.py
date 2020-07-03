# Cast boolean to unsigned int
def condition_to_flag(condition):
    return 1 if condition else 0

# Gets the number of elements in each dimension, and implicitly also how many nested dimensions there are
def get_dimensions(elements):
    dimensions = []
    inner_elements = elements
    while isinstance(inner_elements, list):
        dimensions.append(len(inner_elements))
        inner_elements = inner_elements[0]
    dimensions.reverse()
    return dimensions

def flatten_one_dimension(elements):
    return [lowerer_rank for middle_rank in elements for lowerer_rank in middle_rank]

# Unrolls nested lists into one flat list
def flatten_dimensions(elements, number_of_dimensions=None):
    dimensions = get_dimensions(elements)
    if number_of_dimensions == None:
        number_of_dimensions = len(dimensions)
    for dimension_index in range(0, number_of_dimensions-1):
        elements = flatten_one_dimension(elements)
    return (dimensions[-number_of_dimensions:], elements)

def structure_one_dimension(dimension, elements):
    return [elements[offset:offset+dimension] for offset in range(0, len(elements), dimension)]

# Rolls a flat list back into nested lists
def structure_dimensions(dimensions, elements):
    for dimension_index in range(0, len(dimensions)-1):
        elements = structure_one_dimension(dimensions[dimension_index], elements)
    return elements

def flatten_and_structure_dimensions(op, parameters, number_of_dimensions=None):
    multi_args = isinstance(parameters, tuple)
    dimensions_and_arguments = map(lambda parameter: flatten_dimensions(parameter, number_of_dimensions), parameters if multi_args else [parameters])
    arguments = map(lambda (dimensions, argument): argument, dimensions_and_arguments)
    interleaved_arguments = zip(*arguments) if multi_args else arguments[0]
    processed_elements = op(interleaved_arguments)
    return structure_dimensions(dimensions_and_arguments[0][0], processed_elements)
