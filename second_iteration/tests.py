import cmath
import utils
import primitives
import algorithms
import unittest

class Tests(unittest.TestCase):
    def assertAlmostEqual(self, a, b, tollerance=0.01):
        (dimensions_a, flat_a) = utils.flatten_dimensions(a)
        (dimensions_b, flat_b) = utils.flatten_dimensions(b)
        self.assertEqual(dimensions_a, dimensions_b)
        self.assertTrue(all((abs(flat_a[i]) < tollerance) if flat_b[i] == 0.0 else (abs(flat_a[i]/flat_b[i]-1.0) < tollerance) for i in range(0, len(flat_a))))

    def test_per_element(self):
        factor = 2
        op = lambda (element): element*factor
        elements = [6, 3, 1]
        processed_elements = [12, 6, 2]
        self.assertEqual(primitives.elementwise(op, (elements)), processed_elements)

    def test_elementwise(self):
        combination_op = lambda (a, b): a*b
        elements_a = [[6, 3, 1], [4, 7, 5]]
        elements_b = [[1, 5, 9], [8, 2, 0]]
        product = [[6, 15, 9], [32, 14, 0]]
        self.assertEqual(primitives.elementwise(combination_op, (elements_a, elements_b)), product)

    def test_gather(self):
        indices = [2, 1]
        elements = [5, 3, 7, 2, 0]
        gathered_elements = [7, 3]
        self.assertEqual(primitives.gather(indices, elements), gathered_elements)

    def test_scatter(self):
        indices = [0, 1, 0, 0, 0]
        elements = [5, 3, 7, 2, 0]
        condition_flags = [0, 1, 1, 0, 0]
        scatter_elements = [7, 3]
        result_size = len(scatter_elements)
        self.assertEqual(primitives.scatter(indices, elements, condition_flags, result_size), scatter_elements)

    def test_reduction(self):
        elements = [4, 3, 1, 0, 2]
        accumulator_in = 5
        accumulator_out = 15
        self.assertEqual(primitives.reduction(lambda (a, b): a+b, elements, accumulator_in), accumulator_out)

    def test_exclusive_scan(self):
        elements = [4, 3, 1, 0, 2]
        accumulator_in = 5
        accumulator_out = 15
        integral = [5, 9, 12, 13, 13]
        self.assertEqual(primitives.integral(-1, lambda (a, b): a+b, elements, accumulator_in), (integral, accumulator_out))

    def test_inclusive_scan(self):
        elements = [4, 3, 1, 0, 2]
        accumulator_in = 5
        accumulator_out = 15
        integral = [9, 12, 13, 13, 15]
        self.assertEqual(primitives.integral(1, lambda (a, b): a+b, elements, accumulator_in), (integral, accumulator_out))

    def test_derivative(self):
        elements = [0, 0, 0, 1, 1, 2, 2, 2, 5, 5]
        differences = [-5, 0, 0, 1, 0, 1, 0, 0, 3, 0]
        self.assertEqual(algorithms.derivative(-1, lambda (a, b): b-a, elements), differences)

    def test_multi_dimensional(self):
        op = lambda elements: primitives.integral(1, lambda (a, b): a+b, elements, 0)[0]
        elements = [
            [3, 8, 4, 1],
            [5, 6, 2, 9],
            [1, 4, 7, 0]
        ]
        multi_dimensional_integral = [
            [3, 11, 15, 16],
            [8, 22, 28, 38],
            [9, 27, 40, 50]
        ]
        self.assertEqual(algorithms.multi_dimensional(op, elements), multi_dimensional_integral)

    def test_multi_dimensional_pairs(self):
        number_of_dimensions = 2
        def op(elements):
            keys = map(lambda e: e[0], elements)
            values = map(lambda e: e[1], elements)
            return zip(keys, primitives.integral(1, lambda (a, b): a+b, values, 0)[0])
        elements = [
            [[0, 3], [1, 8], [2, 4], [3, 1]],
            [[4, 5], [5, 6], [6, 2], [7, 9]],
            [[8, 1], [9, 4], [10, 7], [11, 0]]
        ]
        multi_dimensional_integral = [
            [(0, 3), (1, 11), (2, 15), (3, 16)],
            [(4, 8), (5, 22), (6, 28), (7, 38)],
            [(8, 9), (9, 27), (10, 40), (11, 50)]
        ]
        self.assertEqual(algorithms.multi_dimensional(op, elements, number_of_dimensions), multi_dimensional_integral)

    def test_element_index_to_ranks(self):
        dimensions = [2, 3, 4]
        index = 14
        ranks = [0, 1, 2]
        self.assertEqual(algorithms.element_index_to_ranks(dimensions, index), ranks)

    def test_element_ranks_to_index(self):
        dimensions = [2, 3, 4]
        ranks = [0, 1, 2]
        index = 14
        self.assertEqual(algorithms.element_ranks_to_index(dimensions, ranks), index)

    def test_src_transposition(self):
        rank_permutation = [1, 0]
        elements = [[0, 1, 2], [3, 4, 5]]
        transposed_elements = [[0, 3], [1, 4], [2, 5]]
        self.assertEqual(algorithms.src_transposition(rank_permutation, elements), transposed_elements)

    def test_dst_transposition(self):
        rank_permutation = [1, 2, 0]
        elements = [
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
            [[12, 13, 14], [15, 16, 17]],
            [[18, 19, 20], [21, 22, 23]]
        ]
        transposed_elements = [
            [[0, 6, 12, 18], [1, 7, 13, 19], [2, 8, 14, 20]],
            [[3, 9, 15, 21], [4, 10, 16, 22], [5, 11, 17, 23]]
        ]
        self.assertEqual(algorithms.dst_transposition(rank_permutation, elements), transposed_elements)

    def test_inner_product(self):
        combination_op = lambda (a, b): a*b
        reduction_op = lambda (a, b): a+b
        accumulator = 0
        elements_a = [[6, 3, 1], [4, 7, 5]]
        elements_b = [[1, 5, 9], [8, 2, 0]]
        product = 76
        self.assertEqual(algorithms.inner_product(combination_op, reduction_op, accumulator, elements_a, elements_b), product)

    def test_outer_product(self):
        combination_op = lambda (a, b): a*b
        elements_a = [[6, 3, 1], [4, 7, 5]]
        elements_b = [[1, 5, 9], [8, 2, 0]]
        product = [
            [
                [[6, 30, 54], [48, 12, 0]],
                [[3, 15, 27], [24, 6, 0]],
                [[1, 5, 9], [8, 2, 0]]
            ],
            [
                [[4, 20, 36], [32, 8, 0]],
                [[7, 35, 63], [56, 14, 0]],
                [[5, 25, 45], [40, 10, 0]]
            ]
        ]
        self.assertEqual(algorithms.outer_product(combination_op, elements_a, elements_b), product)

    def test_dot_product(self):
        vector_a = [6, 3, 1]
        vector_b = [1, 5, 9]
        scalar_product = 30
        self.assertEqual(algorithms.dot_product(vector_a, vector_b), scalar_product)

    def test_matrix_multiplication(self):
        matrix_a = [[6, 3, 1], [4, 7, 5]]
        matrix_b = [[1, 8], [5, 2], [9, 0]]
        matrix_product = [[30, 54], [84, 46]]
        self.assertEqual(algorithms.matrix_multiplication(matrix_a, matrix_b), matrix_product)

    def test_vector_transformation(self):
        matrix_a = [[6, 3, 1], [4, 7, 5]]
        vector_b = [2, 1, 0]
        transformed_vector = [15, 15]
        self.assertEqual(algorithms.vector_transformation(matrix_a, vector_b), transformed_vector)

    def test_filter(self):
        keep_element_flags = [0, 1, 1, 0, 0]
        elements = [5, 3, 7, 2, 0]
        number_of_elements_to_keep = 2
        elements_to_keep = [3, 7]
        self.assertEqual(algorithms.filter(keep_element_flags, elements), (number_of_elements_to_keep, elements_to_keep))

    def test_lsb_radix_sort(self):
        elements = [[5, 0], [3, 1], [7, 2], [2, 3], [0, 4]]
        sorted_elements = [[0, 4], [2, 3], [3, 1], [5, 0], [7, 2]]
        self.assertEqual(algorithms.lsb_radix_sort(4, 2, elements), sorted_elements)

    def test_partition_edges(self):
        max_key = 5
        elements = [0, 0, 0, 1, 1, 2, 2, 2, 5, 5]
        begin_edge_flags = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
        end_edge_flags = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
        begin_index_per_partition = [0, 3, 5, 0, 0, 8]
        end_index_per_partition = [3, 5, 8, 0, 0, 10]
        self.assertEqual(algorithms.partition_edges(max_key+1, elements), (begin_edge_flags, end_edge_flags, begin_index_per_partition, end_index_per_partition))

    def test_partitioned_counting(self):
        max_key = 5
        elements = [0, 0, 0, 1, 1, 2, 2, 2, 5, 5]
        number_of_elements_per_partition = [3, 2, 3, 0, 0, 2]
        self.assertEqual(algorithms.partitioned_counting(max_key+1, elements), number_of_elements_per_partition)

    def test_partitioned_indices(self):
        max_key = 5
        elements = [0, 0, 0, 1, 1, 2, 2, 2, 5, 5]
        index_of_each_element_in_its_partition = [0, 1, 2, 0, 1, 0, 1, 2, 0, 1]
        self.assertEqual(algorithms.partitioned_indices(max_key+1, elements), index_of_each_element_in_its_partition)

    def test_partitioned_binning(self):
        max_key = 5
        elements = [[0, 3], [0, 2], [0, 1], [1, 2], [1, 1], [2, 3], [2, 2], [2, 1], [5, 2], [5, 1]]
        bins = [6, 3, 6, 0, 0, 3]
        self.assertEqual(algorithms.partitioned_binning(max_key+1, elements), bins)

    def test_fft(self):
        size = 8
        frequency = 2
        amplitude = 1
        wave_length = float(size)/frequency
        time_domain = [cmath.exp(cmath.pi*2.0j*(i/wave_length))*amplitude for i in range(0, size)] # [1, 1j, -1, -1j, 1, 1j, -1, -1j]
        frequency_domain = [0]*size
        frequency_domain[frequency] = amplitude
        self.assertAlmostEqual(algorithms.fft(time_domain, True), frequency_domain)

    def test_convolution(self):
        elements_a = [1, 0, -1, 0, 1, 0, -1, 0]
        elements_b = [2, 1, 0, 0, 0, 0, 0, 1]
        folded = [2, 0, -2, 0, 2, 0, -2, 0]
        self.assertAlmostEqual(algorithms.convolution(elements_a, elements_b), folded)

if __name__ == '__main__':
    unittest.main()
