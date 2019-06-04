import logging
import json
from itertools import groupby, chain
from typing import Union, Tuple, Callable, Iterable, Iterator, Any, List, Dict
import io

logger = logging.getLogger(__name__)


class Graph(object):
    """Class that allows you to create and run computational graphs."""

    def __init__(self, source: Union[str, 'Graph'], name: str = None) -> None:
        """
        Constructs a new 'Graph' object.
        :param source: 1) other Graph object or  2) string with name of a text stream (json)
        or 3) string with name of iterable dict object
        :param name: name of building graph
        :return: returns nothing
        """

        self._parent_graphs = []
        try:
            if isinstance(source, Graph):
                self._source_is_graph = True
                source._count_of_calls += 1
                source._remained_count_of_calls += 1
                self._parent_graphs.append(source)
            elif isinstance(source, str):
                self._source_is_graph = False
            else:
                raise ValueError("Input parameter 'source' can be a graph or a string with input"
                                 " stream name (of json text file or iterable dict object)")
        except Exception as e:
            logger.exception(e)

        self._source = source
        self._name = name

        self._count_of_calls = 0
        self._remained_count_of_calls = 0
        self._output = None
        self._nodes = []
        self._cache = []
        self._is_checked_in_sort = False

    def map(self, mapper: Callable) -> None:
        """
        Constructs a new 'Map' object and adds it to nodes list.
        :param mapper: mapper function
        :return: returns nothing
        """
        new_map_node = Map(mapper)
        self._nodes.append(new_map_node)

    def sort(self, *columns: str, reverse: bool = False) -> None:
        """
        Constructs a new 'Sort' object and adds it to nodes list.
        :param columns: names of columns by which the data will be sorted.
        Data will be sorted by the first column, then by the second column, ... and so on.
        :param reverse: False if you need increase sort, True otherwise
        :return: returns nothing
        """
        new_sort_node = Sort(*columns, reverse=reverse)
        self._nodes.append(new_sort_node)

    def fold(self, folder: Callable, initial_state: Any) -> None:
        """
        Constructs a new 'Fold' object and adds it to nodes list.
        :param folder: folder function
        :param initial_state: initial state for folding
        :return: returns nothing
        """
        new_fold_node = Fold(folder, initial_state)
        self._nodes.append(new_fold_node)

    def reduce(self, reducer: Callable, key: Union[str, Tuple[str, str]]) -> None:
        """
        Constructs a new 'Reduce' object and adds it to nodes list.
        :param reducer: reducer function
        :param key: name of column or tuple of column names, which uses in reducer as a key
        :return: returns nothing
        """
        new_reduce_node = Reduce(reducer, key)
        self._nodes.append(new_reduce_node)

    def join(self, on: 'Graph', strategy: str, key: Union[str, Tuple[str, str]] = None) -> None:
        """
        Constructs a new 'Join' object and adds it to nodes list.
        :param on: 'Graph' object, which is right in join operation
        :param strategy: name of join strategy ('left', 'right', 'inner', 'cross')
        :param key: is None for cross join, otherwise it is a tuple (length = 2) with names
        of columns that participate in join. First name in tuple is a name of column in
        left graph (from which you calls join), second name in tuple is a name of column
        in right graph (which is passed as parameter 'on')
        :return: returns nothing
        """
        on._remained_count_of_calls += 1
        on._count_of_calls += 1
        new_join_node = Join(on, strategy, key)
        self._nodes.append(new_join_node)
        self._parent_graphs.append(on)

    def _internal_run(self, **kwargs: Any) -> None:
        """
        Calls internal run for graphs in topological sort
        :param kwargs: parameters, used for internal run
        :return: returns nothing
        """
        logger.info("Graph {} is running".format(self._name))
        if self._source_is_graph:
            if self._source._count_of_calls > 1:
                node_output = self._source.get_generator_from_output_cache()
                self._source._remained_count_of_calls -= 1
            else:
                node_output = self._source._output
        else:
            try:
                if isinstance(kwargs[self._source], io.TextIOWrapper):
                    node_output = json_to_generator(kwargs[self._source])
                elif isinstance(kwargs[self._source], Iterator):
                    node_output = kwargs[self._source]
                elif isinstance(kwargs[self._source], Iterable):
                    node_output = get_generator_from_iterable(kwargs[self._source])
                else:
                    raise ValueError("Incorrect source format")
            except Exception as e:
                logger.exception(e)
                raise

        for node in self._nodes:
            node_output = node.run(node_output)
        self._output = node_output

        if self._count_of_calls > 1:
            for record in node_output:
                self._cache.append(record)
            self._output = None

        logger.info("Computation of graph {} is done".format(self._name))

    def _topological_sort(self) -> List['Graph']:
        """
        Makes topological sort of graphs by deep first search algorithm
        :return: returns nothing
        """
        graphs_in_topological_order = [self]
        stack_with_graphs = [self]
        self._is_checked_in_sort = True
        while len(stack_with_graphs) > 0:
            current_graph = stack_with_graphs.pop()
            for parent_graph in current_graph._parent_graphs:
                if not parent_graph._is_checked_in_sort:
                    parent_graph._is_checked_in_sort = True
                    stack_with_graphs.append(parent_graph)
                    graphs_in_topological_order.append(parent_graph)
        graphs_in_topological_order.reverse()
        return graphs_in_topological_order

    def get_generator_from_output_cache(self) -> Iterator:
        for record in self._cache:
            yield record

    def run(self, output_stream: io.TextIOWrapper = None, verbose: bool = False, **kwargs) -> List[Dict]:
        """
        Runs graph
        :param output_stream: opened file for writing
        :param verbose: parameter for logging. True sets INFO logging level, False sets ERROR logging level
        :param kwargs: dict of all input arguments for graph running
        :return: list of dicts, result of computing graph
        """
        if verbose:
            logger.setLevel(logging.INFO)

        graphs_in_topological_order = self._topological_sort()
        for graph_to_run in graphs_in_topological_order:
            graph_to_run._internal_run(**kwargs)

        if output_stream is not None:
            for record in self._output:
                output_stream.write(json.dumps(record) + "\n")
            return []
        else:
            return list(self._output)


class Map(object):
    """Class represents Map node in Graph"""

    def __init__(self, mapper: Callable) -> None:
        """
        Constructs a new 'Map' object
        :param mapper: mapper function
        :return: returns nothing
        """
        self._mapper = mapper

    def run(self, input_generator: Iterator) -> Iterator:
        """
        Runs map node
        :param input_generator: input for map node
        :return: returns generator, result of application mapper function
        """
        for record in input_generator:
            yield from self._mapper(record)


def json_to_generator(input_stream: io.TextIOWrapper) -> Iterator:
    """
    Transforms json text stream into generator with dict values
    :param input_stream: input text stream
    :return: returns generator with json transformed to dict
    """
    for line in input_stream:
        yield json.loads(line)


def get_generator_from_iterable(iterable_value: Iterable) -> Iterator:
    for record in iterable_value:
        yield record


class Sort(object):
    """Class represents Sort node in Graph"""

    def __init__(self, *columns: str, reverse: bool = False) -> None:
        """
        Constructs a new 'Sort' object
        :param columns: names of columns by which the data will be sorted.
        Data will be sorted by the first column, then by the second column, ... and so on.
        :param reverse: False if you need increase sort, True otherwise
        :return: returns nothing
        """
        self._columns = columns
        self._data = []
        self._reverse = reverse

    def run(self, input_generator: Iterator) -> Iterator:
        """
        Runs Sort node
        :param input_generator: input for sort node
        :return: returns generator, result of sort
        """
        for record in input_generator:
            self._data.append(record)
        self._data.sort(key=lambda record: tuple(record[column] for column in self._columns), reverse=self._reverse)

        # print(self._data)
        for record in self._data:
            yield record

        self._data = []


class Fold(object):
    """Class represents Fold node in Graph"""

    def __init__(self, folder: Callable, initial_state: Dict) -> None:
        """
        Constructs a new 'Fold' object
        :param folder: folder function
        :param initial_state: initial state for folding
        :return: returns nothing
        """
        self._folder = folder
        self._initial_state = initial_state

    def run(self, input_generator: Iterator) -> Iterator:
        """
        Runs Fold node
        :param input_generator: input for fold node
        :return: returns generator, result of application folder function
        """
        state = self._initial_state
        for record in input_generator:
            state = self._folder(state, record)
        yield state


class Reduce(object):
    """Class represents Reduce node in Graph"""

    def __init__(self, reducer: Callable, key: Union[str, Tuple[str, str]]) -> None:
        """
        Constructs a new 'Reduce' object
        :param reducer: reducer function
        :param key: name of column or tuple of column names, which uses in reducer as a key
        :return: returns nothing
        """
        self._reducer = reducer
        if isinstance(key, str):
            self._key = [key]
        else:
            self._key = key

    def run(self, input_generator: Iterable) -> Iterator:
        """
        Runs Reduce node
        :param input_generator: input for reduce node
        :return: returns generator, result of application reducer function
        """
        for _, group in groupby(input_generator,
                                key=lambda record: tuple(record[column] for column in self._key)):
            yield from self._reducer(group)


class Join(object):
    """Class represents Join node in Graph"""

    def __init__(self, on: Graph, strategy: str, key: Union[str, Tuple[str, str]] = None) -> None:
        """
        Constructs a new 'Join' object.
        :param on: 'Graph' object, which is right in join operation
        :param strategy: name of join strategy ('left', 'right', 'inner', 'cross')
        :param key: is None for cross join, otherwise it is a tuple (length = 2) with names
        of columns, which participate in join. First name in tuple is a name of column in
        left graph (from which you calls join), second name in tuple is a name of column
        in right graph (which is passed as parameter 'on')
        :return: returns nothing
        """
        self._strategy = strategy
        if strategy != 'cross':
            if isinstance(key, str):
                self._key = tuple([key, key])
            else:
                self._key = key

        else:
            self._key = None
        self._on = on
        self._keys_with_common_name = []
        self._left_table = None
        self._right_table = None
        self._right_table_keys = []
        self._left_table_keys = []

    def _join_reducer(self, records: Iterable) -> Iterator:
        """
        Reducer for _join function
        :param records: records in group
        :return: yiels records from one group
        """
        yield records

    def _stop_join_iteration_condition(self, left_table_part: Iterable, right_table_part: Iterable) -> bool:
        """
        Returns stop join iteration condition for _join function
        :param left_table_part: list with dicts of left table part
        :param right_table_part: list with dicts of right table part
        :return: returns bool. True if need to stop iteration, False otherwise
        """
        if self._strategy == "inner":
            return len(left_table_part) == 0 or len(right_table_part) == 0
        elif self._strategy == "left":
            return len(left_table_part) == 0
        elif self._strategy == "right":
            return len(right_table_part) == 0
        else:
            return len(left_table_part) == 0 and len(right_table_part) == 0

    def _cartesian_product_left_non_empty_right_empty(self, left_table_part: Iterable) -> Iterator:
        """
        Yield records that are rows in cartesian product of non-empty left and empty right tables.
        :param left_table_part: list with dicts of left table part
        :return: yield records that are rows in cartesian product
        """
        new_record_right = {}
        for column_name_right in self._right_table_keys:
            if column_name_right in self._keys_with_common_name:
                new_record_right.update({column_name_right + '_right': None})
            else:
                new_record_right.update({column_name_right: None})
        for record_left in left_table_part:
            new_record = {}
            for column_name_left in record_left:
                if column_name_left in self._keys_with_common_name:
                    new_record.update({column_name_left + '_left': record_left[column_name_left]})
                else:
                    new_record.update({column_name_left: record_left[column_name_left]})
            new_record.update(new_record_right)
            yield new_record

    def _cartesian_product_left_empty_right_non_empty(self, right_table_part: Iterable) -> Iterator:
        """
        Yield records that are rows in cartesian product of empty left and non-empty right tables.
        :param right_table_part: list with dicts of right table part
        :return: yield records that are rows in cartesian product
        """
        new_record_left = {}
        for column_name_left in self._left_table_keys:
            if column_name_left in self._keys_with_common_name:
                new_record_left.update({column_name_left + '_left': None})
            else:
                new_record_left.update({column_name_left: None})
        for record_right in right_table_part:
            new_record = {}
            for column_name_right in record_right:
                if column_name_right in self._keys_with_common_name:
                    new_record.update({column_name_right + '_right': record_right[column_name_right]})
                else:
                    new_record.update({column_name_right: record_right[column_name_right]})
                new_record.update(new_record_left)
            yield new_record

    def _cartesian_product_left_non_empty_right_non_empty(self, left_table_part: Iterable,
                                                          right_table_part: Iterable) -> Iterator:
        """
        Yield records that are rows in cartesian product of two non-empty table parts
        :param left_table_part: list with dicts of left table part
        :param right_table_part: list with dicts of right table part
        :return: yield records that are rows in cartesian product
        """
        for record_left in left_table_part:
            new_record_left = {}
            for column_name_left in record_left:  # Creates new record from record_left with new keys
                if column_name_left in self._keys_with_common_name:
                    new_record_left.update({column_name_left + '_left': record_left[column_name_left]})
                else:
                    new_record_left.update({column_name_left: record_left[column_name_left]})
            new_record = {}
            new_record.update(new_record_left)
            for record_right in right_table_part:
                for column_name_right in record_right:
                    if column_name_right in self._keys_with_common_name:
                        new_record.update({column_name_right + '_right': record_right[column_name_right]})
                    else:
                        new_record.update({column_name_right: record_right[column_name_right]})
                yield new_record
                new_record = {}
                new_record.update(new_record_left)

    def _get_next_table_part(self, table_groups: Iterator) -> Iterable:
        """
        Make next for iterator with table parts and returns list that is new table part.
        If next(iterator) is None returns empty list
        :param table_groups: iterator with table groups
        :return: list that is next table part
        """
        next_group = next(table_groups, None)
        if next_group is None:
            table_part = []
        else:
            table_part = list(next_group)
        return table_part

    def _join(self) -> None:
        """
        Runs join (inner, left, right or full)
        :return: returns nothing
        """
        sort_node_left = Sort(self._key[0])
        sort_node_right = Sort(self._key[1])
        reduce_node_left = Reduce(self._join_reducer, self._key[0])
        reduce_node_right = Reduce(self._join_reducer, self._key[1])
        left_table_groups = reduce_node_left.run(sort_node_left.run(self._left_table))
        right_table_groups = reduce_node_right.run(sort_node_right.run(self._right_table))
        left_table_part = self._get_next_table_part(left_table_groups)
        right_table_part = self._get_next_table_part(right_table_groups)

        while not self._stop_join_iteration_condition(left_table_part, right_table_part):
            if len(left_table_part) != 0 and len(right_table_part) == 0:
                if self._strategy == "inner" or self._strategy == "right":
                    return
                else:
                    yield from self._cartesian_product_left_non_empty_right_empty(left_table_part)
                    left_table_part = self._get_next_table_part(left_table_groups)

            elif len(left_table_part) == 0 and len(right_table_part) != 0:
                if self._strategy == "inner" or self._strategy == "left":
                    return
                else:
                    yield from self._cartesian_product_left_empty_right_non_empty(right_table_part)
                    right_table_part = self._get_next_table_part(right_table_groups)

            else:
                key_left = left_table_part[0][self._key[0]]
                key_right = right_table_part[0][self._key[1]]

                if key_left == key_right:
                    yield from self._cartesian_product_left_non_empty_right_non_empty(left_table_part, right_table_part)
                    left_table_part = self._get_next_table_part(left_table_groups)
                    right_table_part = self._get_next_table_part(right_table_groups)

                elif key_left < key_right:
                    if self._strategy == "left" or self._strategy == "full":
                        yield from self._cartesian_product_left_non_empty_right_empty(left_table_part)
                    left_table_part = self._get_next_table_part(left_table_groups)

                else:
                    if self._strategy == "right" or self._strategy == "full":
                        yield from self._cartesian_product_left_empty_right_non_empty(right_table_part)
                    right_table_part = self._get_next_table_part(right_table_groups)

    def _cross_join(self):
        """
        Runs cross join
        :return: yield generator, result of join
        """
        left_table = list(self._left_table)
        right_table = list(self._right_table)

        for record_left in left_table:
            new_record_left = {}
            for column_name_left in self._left_table_keys:
                if column_name_left in self._keys_with_common_name:
                    new_record_left.update({column_name_left + '_left': record_left[column_name_left]})
                else:
                    new_record_left.update({column_name_left: record_left[column_name_left]})
            new_record = {}
            new_record.update(new_record_left)
            for record_right in right_table:
                for column_name_right in record_right:
                    if column_name_right in self._keys_with_common_name:
                        new_record.update(
                            {column_name_right + '_right': record_right[column_name_right]})
                    else:
                        new_record.update({column_name_right: record_right[column_name_right]})
                yield new_record
                new_record = {}
                new_record.update(new_record_left)

    def run(self, left_table):
        """
        Runs Join node
        :param left_table: input for join node
        :return: yield from selected join function
        """

        if self._on._count_of_calls == 1:
            self._right_table = self._on._output
        else:
            self._right_table = self._on.get_generator_from_output_cache()

        self._on._remained_count_of_calls -= 1

        self._left_table = left_table

        first_elem_right_table = next(self._right_table, None)
        first_elem_left_table = next(self._left_table, None)

        if first_elem_right_table is None or first_elem_left_table is None:
            return

        self._left_table = chain([first_elem_left_table], self._left_table)
        self._right_table = chain([first_elem_right_table], self._right_table)

        self._left_table_keys = set(first_elem_left_table.keys())
        self._right_table_keys = set(first_elem_right_table.keys())

        self._keys_with_common_name = self._left_table_keys & self._right_table_keys
        strategies_names_require_keys = ['inner', 'left', 'right', 'full']
        try:
            if self._strategy in strategies_names_require_keys:
                if self._key is None:
                    raise ValueError("Input keys for join. Any join strategy (except the cross one)"
                                     "requires to specify keys by which the join occurs")
                yield from self._join()

            elif self._strategy == 'cross':
                yield from self._cross_join()

            else:
                raise ValueError("Incorrect strategy of join {}. Possible variants: left, right, inner,"
                                 " cross, full".format(self._strategy))
        except Exception as e:
            logger.exception(e)

        if self._on._remained_count_of_calls == 0:
            self._on._cache = []
