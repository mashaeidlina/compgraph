import graph
import re
from typing import Iterable, Dict, Iterator
from math import log, acos, sin, cos, radians
from collections import Counter
import time


def extract_words(text: str) -> Iterator:
    """
    Extracts words from text
    :param text: Text with words
    :return: generator with extracted words
    """
    delimiters = [
        ' ', '.', '?', '!', ':', ',', '-', '"',
        ';', '$', '%', '^', '&', '*', '(', ')',
        '@', '#', '~', '<', '>', '/', '\n',
        '[', ']'
    ]
    regex_pattern = '|'.join(map(re.escape, delimiters))
    for word in re.split(regex_pattern, text):
        if word:
            yield word.lower().strip()


def build_word_count_graph(input_stream: str, text_column: str = 'text', count_column: str = 'count') -> graph.Graph:
    """
    Builds graph which counts words in a collection of documents. Documents represents rows
    in the following format (file: examples/resources/text_corpus.txt):
    {'doc_id': 'name', 'text':'...'}, where text_column = 'text'.

    The result looks like this: {'count': count, 'text': word}, where count is a number of
    occurrences of word in all texts in total (here count_column = 'count')

    :param input_stream: name of input stream:
        1) string with name of a text stream (json) or
        2) string with name of iterable (list) with dicts
        3) generator that produces dicts
    :param text_column: name of column with texts in input stream
    :param count_column: name of column with counts
    :return: returns 'Graph' object
    """

    def emit_words(record: Dict) -> Iterator:
        """
        Mapper. Builds dicts with 2 fields: count_column with value = 1 and text_column
        with word from record (word is extracted by extract_words function)

        From one record with string this function yields lots of dicts,
        which number is equal to number of words in a string

        :param record: record with one word in text_column
        :return: yield dicts with 2 fields
        """
        for word in extract_words(record[text_column]):
            yield {count_column: 1, text_column: word}

    def collect_counts(records: Iterable) -> Iterator:
        """
        Reducer. Counts words
        :param records: an iterable with dicts. All dicts have equal value
        of field text_column (they represent the same word)
        :return: yield dict with word and number of this word
        """
        count_records = 0
        for record in records:
            count_records += 1
        yield {count_column: count_records, text_column: record[text_column]}

    my_graph = graph.Graph(source=input_stream)
    my_graph.map(emit_words)
    my_graph.sort(text_column)
    my_graph.reduce(collect_counts, key=text_column)
    my_graph.sort(count_column)
    return my_graph


def build_inverted_index_graph(input_stream: str, doc_column: str = 'doc_id', text_column: str = 'text') -> graph.Graph:
    """
    Builds graph which constructs an inverted index.
    As input this graph uses a collection of documents in the following format
    (example in file: examples/resources/text_corpus.txt):
    {'doc_id': 'name', 'text':'...'}, where text_column = 'text'.
    Inverted index is a data structure that for each word stores a list of documents in which it occurs,
    sorted in order of relevance. Relevance is considered by the metric tf-idf.
    For each pair (word, document) tf-idf is calculated as follows:
    TFIDF(word_i, doc_i) = (frequency of word_i in doc_i ) *
        *log ((total number of docs) / (docs where word_i is present)) = tf * idf
    For each word it is necessary to calculate the top 3 documents by tf-idf.

    The result looks like this: {"text": "hello",  "doc_id": 5, "tf_idf": 0.2703}

    :param input_stream: name of input stream:
        1) string with name of a text stream (json) or
        2) string with name of iterable (list) with dicts
        3) generator that produces dicts
    :param doc_column: name of column with document id
    :param text_column: name of column with text
    :return: returns 'Graph' object
    """

    def emit_words(record):
        """
        Mapper. Builds dicts with 2 fields: count_column with value = 1 and text_column
        with word from record (word is extracted by extract_words function)
        :param record: record with one word in text_column
        :return: yield dicts with 2 fields
        From one record with string this function yields lots of dicts,
        which number is equal to number of words in a string
        """
        for word in extract_words(record[text_column]):
            yield {doc_column: record[doc_column], text_column: word}

    def count_records(state: Dict, _) -> Dict:
        """
        Folder. Counts all records. Final state of fold stage is
        dict with value of 'docs_count' = number of documents
        :param state: current state for folder
        :param _: unused parameter with record
        :return: dict with new state
        """
        state['docs_count'] += 1
        return state

    def unique(records: Iterable) -> Iterator:
        """
        Reducer. From a number of selected records yield only one
        :param records: selected records
        :return: yield unique record
        """
        for record in records:
            yield record
            break

    def calculate_idf(records: Iterable) -> Iterator:
        """
        Reducer. Calculates log in tf-idf formula (idf)
        :param records: selected records
        :return: yield record with idf value
        """
        records_count = 0
        for record in records:
            records_count += 1
        yield {
            text_column: record[text_column],
            'idf': log(record['docs_count'] / records_count)
        }

    def calculate_tf(records: Iterable) -> Iterator:
        """
        Reducer. Calculates tf in tf-idf formula
        :param records: selected records
        :return: yield record with tf value
        """
        word_count = Counter()

        for record in records:
            word_count[record[text_column]] += 1

        total = sum(word_count.values())
        for w, count in word_count.items():
            yield {
                doc_column: record[doc_column],
                text_column: w,
                'tf': count / total
            }

    def invert_index(records: Iterable) -> Iterator:
        """
        Reducer. Calculates tf-idf index and selects top 3 records by tf_idf index
        :param records: selected records
        :return: yield records with tf-idf value
        """
        new_records = []
        for record in records:
            tf_idf = record['tf'] * record['idf']
            record.update({'tf_idf': tf_idf})
            new_records.append(record)
        new_records.sort(key=lambda row: row['tf_idf'], reverse=True)
        counter = 0
        for record in new_records:
            if counter > 2:
                break
            yield {
                text_column: record[text_column + "_left"],
                doc_column: record[doc_column],
                'tf_idf': record['tf_idf']
            }
            counter += 1

    input_graph = graph.Graph(source=input_stream)

    split_word_graph = graph.Graph(source=input_graph)
    split_word_graph.map(emit_words)

    count_docs_graph = graph.Graph(source=input_graph)
    count_docs_graph.fold(count_records, {'docs_count': 0})

    count_idf_graph = graph.Graph(source=split_word_graph)
    count_idf_graph.sort(doc_column, text_column)
    count_idf_graph.reduce(unique, key=(doc_column, text_column))
    count_idf_graph.join(count_docs_graph, strategy='cross')
    count_idf_graph.sort(text_column)
    count_idf_graph.reduce(calculate_idf, key=text_column)

    calc_index_graph = graph.Graph(source=split_word_graph)
    calc_index_graph.sort(doc_column)
    calc_index_graph.reduce(calculate_tf, key=doc_column)
    calc_index_graph.join(count_idf_graph, strategy='left', key=text_column)
    calc_index_graph.sort(text_column + "_left")
    calc_index_graph.reduce(invert_index, key=text_column + "_left")

    return calc_index_graph


def build_pmi_graph(input_stream: str, doc_column: str = 'doc_id', text_column: str = 'text'):
    """
    Builds graph which constructs an inverted index.
    As input this graph uses a collection of documents in the following format
    (example in file: examples/resources/text_corpus.txt):
    {'doc_id': 'name', 'text':'...'}, where text_column = 'text'.

    pmi(word_i, doc_i) = log((frequency of word_i in doc_i) / (frequency of word_i in all documents combined))
    For each text it is neccessary to find the top 10 words by pmi index, each of which is longer
    than four characters and occurs in at least in two documents.

    The result looks like this: {"text": "adjured", "doc_id": "a_tale_of_two_cities", "pmi": 2.914494591458188}

    :param input_stream: name of input stream:
        1) string with name of a text stream (json) or
        2) string with name of iterable (list) with dicts
        3) generator that produces dicts
    :param doc_column: name of column with document id
    :param text_column: name of column with text
    :return: returns 'Graph' object
    """
    def emit_words(record):
        """
        Mapper. Builds dicts with 2 fields: count_column with value = 1 and text_column
        with word from record (word is extracted by extract_words function)
        :param record: record with one word in text_column
        :return: yield dicts with 2 fields
        From one record with string this function yields lots of dicts,
        which number is equal to number of words in a string
        """
        for word in extract_words(record[text_column]):
            yield {doc_column: record[doc_column], text_column: word}

    def count_records(state: Dict, _) -> Dict:
        """
        Folder. Counts all records. Final state of fold stage is
        dict with value of 'docs_count' = number of documents
        :param state: current state for folder
        :param _: unused parameter with record
        :return: returns dict with new state
        """
        state['docs_count'] += 1
        return state

    def count_words(records: Iterable) -> Iterator:
        """
        Reducer. Counts words in all documents
        :param records: records with the same word
        :return: yield record with count of word
        """
        rows_count = 0
        for record in records:
            rows_count += 1
        yield {text_column: record[text_column], 'word_count': rows_count}

    def calculate_frequency_of_word_in_document(records: Iterable) -> Iterator:
        """
        Reducer. Calculates frequency of word in document (nominator in
        pmi formula)
        :param records: record with the same doc_id
        :return: yield record with frequency of every word in document
        """
        word_count = Counter()
        for record in records:
            word_count[record[text_column]] += 1
        total = sum(word_count.values())
        for word, count in word_count.items():
            if count >= 2:
                yield {
                    doc_column: record[doc_column],
                    text_column: word,
                    'no': count / total
                }

    def calculate_frequency_of_word_in_all_documents(record: Dict) -> Iterator:
        """
        Mapper. Calculates frequency of word in all documents combined (denominator
        in pmi formula)
        :param record: current record
        :return: yield record with frequency of word in all documents combined
        """
        yield {text_column: record[text_column], 'dn': record['word_count'] / record['docs_count']}

    def calc_pmi(records: Iterable) -> Iterator:
        """
        Reducer. Calculates pmi index and selects top 10 records by pmi index
        :param records: selected records
        :return: yield record with pmi value
        """
        new_records = []
        for record in records:
            pmi = log(record['no'] / record['dn'])
            record.update({'pmi': pmi})
            new_records.append(record)
        new_records.sort(key=lambda row: row['pmi'], reverse=True)
        counter = 0
        for record in new_records:
            if counter > 9:
                break
            yield {
                text_column: record[text_column + "_left"],
                doc_column: record[doc_column],
                'pmi': record['pmi']
            }
            counter += 1

    split_word_graph = graph.Graph(source=input_stream)
    split_word_graph.map(emit_words)

    count_docs_graph = graph.Graph(source=split_word_graph)
    count_docs_graph.fold(count_records, {'docs_count': 0})

    calc_denominator_graph = graph.Graph(source=split_word_graph)
    calc_denominator_graph.sort(text_column)
    calc_denominator_graph.reduce(count_words, key=text_column)
    calc_denominator_graph.join(count_docs_graph, strategy='cross')
    calc_denominator_graph.map(calculate_frequency_of_word_in_all_documents)

    calc_nomenator_graph = graph.Graph(source=split_word_graph)
    calc_nomenator_graph.sort(doc_column)
    calc_nomenator_graph.reduce(calculate_frequency_of_word_in_document, key=doc_column)

    calc_nomenator_graph.join(calc_denominator_graph, strategy='left', key=text_column)
    calc_nomenator_graph.sort(doc_column, doc_column)
    calc_nomenator_graph.reduce(calc_pmi, key=doc_column)
    return calc_nomenator_graph


def build_yandex_maps_graph(input_stream_times: str, input_stream_coords: str) -> graph.Graph:
    """
    Builds a graph which contains average speed in the city from the day of the week and hour.

    The city streets are defined as a graph. All edges have unique id (edge_id).
    The information about movements on streets stored in table in the following format
    (example in file: examples/resources/travel_times.txt):
    {'edge_id':'624','enter_time': '20170912T123410.1794','leave_time': '20170912T123412.68'},
    where edge_id is the edge identifier of the road, enter_time and leave_time are respectively
    the time of entry and exit to this edge (time in utc).

    Also, information about street lengths is in a table with such a format (example
    in examples/resources/graph_data.txt):
    {'edge_id': '313', start': [37.31245, 51.256734], 'end': [37.31245, 51.256734]},
    where start and end are the start and end coordinates of the edge specified in the format,
    ('lon', 'lat'), lon - longitude, lat - latitude.

    The result table looks like this one:
    {'weekday': 'Mon', 'hour': 4, 'speed': 44.812}

    :param input_stream_times: name of input stream with times:
        1) string with name of a text stream (json) or
        2) string with name of iterable (list) with dicts
        3) generator that produces dicts
    :param input_stream_coords: name of input stream with coordinates:
        1) string with name of a text stream (json) or
        2) string with name of iterable (list) with dicts
        3) generator that produces dicts
    :return: returns 'Graph' object
    """
    weeks_dict = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

    def get_week_hour(record: Dict) -> Iterator:
        """
        Mapper. From record in tables with times forms new record with separated weekday and hour.
        Also, result record has a field 'spent_time' = 'leave_time' - 'enter_time' for each record (in seconds),
        and field 'edge_id'.

        The result record looks like this one:
        {'weekday': 'Mon', 'hour': 4, 'spent_time': 44.812}
        :param record: record from table with times
        :return: yield new record with fields:  "weekday", "hour", "spent_time", "edge_id"
        """

        leave_time_str = record["leave_time"]
        leave_struct_time = time.strptime(leave_time_str[0:15], "%Y%m%dT%H%M%S")
        weekday = weeks_dict[leave_struct_time.tm_wday]
        hour = leave_struct_time.tm_hour
        enter_time_str = record["enter_time"]
        enter_struct_time = time.strptime(enter_time_str[0:15], "%Y%m%dT%H%M%S")
        secs_leave = leave_time_str[15:]
        if len(secs_leave) == 0:
            secs_leave = 0
        else:
            secs_leave = float(secs_leave)
        secs_enter = enter_time_str[15:]
        if len(secs_enter) == 0:
            secs_enter = 0
        else:
            secs_enter = float(secs_enter)
        spent_time = (time.mktime(leave_struct_time) + secs_leave -
                      time.mktime(enter_struct_time) - secs_enter)
        yield {
            "weekday": weekday,
            "hour": hour,
            "spent_time": spent_time,
            "edge_id": record["edge_id"]
        }

    def get_lengths(record: Dict) -> Iterator:
        """
        Mapper. From record in tables with coordinates forms new record with length of street,
        using formula:
        length = R * arccos(sin(lon1) * sin(lon2) + cos(lon1) * cos(lon2) * cos(lat2 - lat1)),
        where R - Earth radius, lon1, lat1 - longitude and latitude of first coordinate
        (start coordinate), lon2, lat2 - longitude and latitude of second coordinate
        (end coordinate), length - length of street with edge_id in meters.
        R = 6371302 (almost)

        The result record looks like this one:
        {'edge_id': 313, 'length': 100}

        :param record: record from table with coordinates
        :return: yield new record with fields: "edge_id", "length"
        """

        start_coords = record["start"]
        end_coords = record["end"]
        length = 6371302 * acos(sin(radians(start_coords[0])) * sin(radians(end_coords[0]))
                                + cos(radians(start_coords[0])) * cos(radians(end_coords[0]))
                                * cos(radians(end_coords[1]) - radians(start_coords[1])))
        yield {"edge_id": record["edge_id"], "length": length}

    def get_speeds(record: Dict) -> Iterator:
        """
        Mapper. Forms new record from record with calculated fields "length" and "spent_time"
        by adding new field "speed" = "length" / "spent_time". "speed" is a
        speed in meters per second
        :param record: record with "length" and "spent_time"
        :return: yield updated record with "speed"
        """
        record.update({"speed": record["length"] / record["spent_time"]})
        yield record

    def get_average_speed(records: Iterable) -> Iterator:
        """
        Forms mew record with average speed (in meters per second) and converts it
        in kilometers per hour.
        :param records: table with equal field "weekday" and "hour"
        :return: yield new record with fields "weekday", "hour" and "speed"
        """
        sum_speeds = 0
        counter = 0
        for record in records:
            sum_speeds += record["speed"]
            counter += 1
        yield {
            "weekday": record["weekday"],
            "hour": record["hour"],
            "speed": (18 * sum_speeds) / (5 * counter)
        }

    times_graph = graph.Graph(source=input_stream_times)
    times_graph.map(get_week_hour)

    lengths_graph = graph.Graph(source=input_stream_coords)
    lengths_graph.map(get_lengths)

    times_graph.join(on=lengths_graph, key="edge_id", strategy="left")
    times_graph.map(get_speeds)
    times_graph.sort("weekday", "hour")
    times_graph.reduce(get_average_speed, key=("weekday", "hour"))
    return times_graph
