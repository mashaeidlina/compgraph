from tests import algorithms
from graph import Graph
from itertools import cycle, islice
from pytest import approx
from collections import Counter


def sorted_eq(tb1, tb2, key):
    return sorted(tb1, key=lambda x: tuple(x[k] for k in key)) == \
           sorted(tb2, key=lambda x: tuple(x[k] for k in key))


def test_word_count():
    docs = [
        {'doc_id': 1, 'text': 'hello, my little WORLD'},
        {'doc_id': 2, 'text': 'Hello, my little little hell'}
    ]

    etalon = [
        {'count': 1, 'text': 'hell'},
        {'count': 1, 'text': 'world'},
        {'count': 2, 'text': 'hello'},
        {'count': 2, 'text': 'my'},
        {'count': 3, 'text': 'little'}
    ]

    g = algorithms.build_word_count_graph('docs')

    result = g.run(docs=docs)

    assert result == etalon


def test_word_count_multiple_call():
    g = algorithms.build_word_count_graph('text')

    rows1 = [
        {'doc_id': 1, 'text': 'hello, my little WORLD'},
    ]

    etalon1 = [
        {'count': 1, 'text': 'world'},
        {'count': 1, 'text': 'hello'},
        {'count': 1, 'text': 'my'},
        {'count': 1, 'text': 'little'}
    ]

    result1 = g.run(
        text=rows1
    )

    rows2 = [
        {'doc_id': 1, 'text': 'hello, my little WORLD'},
        {'doc_id': 2, 'text': 'Hello, my little little hell'}
    ]

    etalon2 = [
        {'count': 1, 'text': 'hell'},
        {'count': 1, 'text': 'world'},
        {'count': 2, 'text': 'hello'},
        {'count': 2, 'text': 'my'},
        {'count': 3, 'text': 'little'}
    ]

    result2 = g.run(
        text=rows2,
        verbose=True
    )
    print(etalon2)
    print(result2)
    assert sorted_eq(etalon1, result1, ['text'])
    assert sorted_eq(etalon2, result2, ['text'])


def test_tf_idf():
    rows = [
        {'doc_id': 1, 'text': 'hello, little world'},
        {'doc_id': 2, 'text': 'little'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'little? hello little world'},
        {'doc_id': 5, 'text': 'HELLO HELLO! WORLD...'},
        {'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!!'}
    ]

    etalon = [
        {"text": "hello",  "doc_id": 5, "tf_idf": approx(0.2703, 0.001)},
        {"text": "hello", "doc_id": 1, "tf_idf": approx(0.1351, 0.001)},
        {"text": "hello", "doc_id": 4, "tf_idf": approx(0.1013, 0.001)},
        {"text": "little", "doc_id": 2, "tf_idf": approx(0.4054, 0.001)},
        {"text": "little", "doc_id": 3, "tf_idf": approx(0.4054, 0.001)},
        {"text": "little", "doc_id": 4, "tf_idf": approx(0.2027, 0.001)},
        {"text": "world", "doc_id": 6, "tf_idf": approx(0.3243, 0.001)},
        {"text": "world", "doc_id": 1, "tf_idf": approx(0.1351, 0.001)},
        {"text": "world", "doc_id": 5, "tf_idf": approx(0.1351, 0.001)}
    ]

    g = algorithms.build_inverted_index_graph('texts')
    result = g.run(texts=rows)

    assert result == etalon


def test_pmi():
    rows = [
        {'doc_id': 1, 'text': 'hello, little world'},
        {'doc_id': 2, 'text': 'little'},
        {'doc_id': 3, 'text': 'little little little'},
        {'doc_id': 4, 'text': 'little? hello little world'},
        {'doc_id': 5, 'text': 'HELLO HELLO! WORLD...'},
        {'doc_id': 6, 'text': 'world? world... world!!! WORLD!!! HELLO!!! HELLO!!!!!!!'}
    ]

    etalon = [
        {'text': 'little', 'doc_id': 3, 'pmi': approx(1.0498, 0.001)},
        {'text': 'little', 'doc_id': 4, 'pmi': approx(0.3567, 0.001)},
        {'text': 'hello', 'doc_id': 5, 'pmi': approx(0.7985, 0.001)},
        {'text': 'world', 'doc_id': 6, 'pmi': approx(0.6444, 0.001)},
        {'text': 'hello', 'doc_id': 6, 'pmi': approx(0.1054, 0.001)}
    ]

    g = algorithms.build_pmi_graph('texts')   # must work with iterable
    result = g.run(texts=iter(rows))

    assert etalon == result


def test_yandex_maps():
    lengths = [
        {"start": [37.84870228730142, 55.73853974696249], "end": [37.8490418381989, 55.73832445777953],
         "edge_id": 8414926848168493057},
        {"start": [37.524768467992544, 55.88785375468433], "end": [37.52415172755718, 55.88807155843824],
         "edge_id": 5342768494149337085},
        {"start": [37.56963176652789, 55.846845586784184], "end": [37.57018438540399, 55.8469259692356],
         "edge_id": 5123042926973124604},
        {"start": [37.41463478654623, 55.654487907886505], "end": [37.41442892700434, 55.654839486815035],
         "edge_id": 5726148664276615162},
        {"start": [37.584684155881405, 55.78285809606314], "end": [37.58415022864938, 55.78177368734032],
         "edge_id": 451916977441439743},
        {"start": [37.736429711803794, 55.62696328852326], "end": [37.736344216391444, 55.626937723718584],
         "edge_id": 7639557040160407543},
        {"start": [37.83196756616235, 55.76662947423756], "end": [37.83191015012562, 55.766647034324706],
         "edge_id": 1293255682152955894},
    ]

    times = [
        {"leave_time": "20171020T112238.723000",
         "enter_time": "20171020T112237.427000", "edge_id": 8414926848168493057},
        {"leave_time": "20171011T145553.040000",
         "enter_time": "20171011T145551.957000", "edge_id": 8414926848168493057},
        {"leave_time": "20171020T090548.939000",
         "enter_time": "20171020T090547.463000", "edge_id": 8414926848168493057},
        {"leave_time": "20171024T144101.879000",
         "enter_time": "20171024T144059.102000", "edge_id": 8414926848168493057},
        {"leave_time": "20171022T131828.330000",
         "enter_time": "20171022T131820.842000", "edge_id": 5342768494149337085},
        {"leave_time": "20171014T134826.836000",
         "enter_time": "20171014T134825.215000", "edge_id": 5342768494149337085},
        {"leave_time": "20171010T060609.897000",
         "enter_time": "20171010T060608.344000", "edge_id": 5342768494149337085},
        {"leave_time": "20171027T082600.201000",
         "enter_time": "20171027T082557.571000", "edge_id": 5342768494149337085}
    ]

    etalon = [
        {'weekday': 'Fri', 'hour': 8, 'speed': approx(97.4886, 0.001)},
        {'weekday': 'Fri', 'hour': 9, 'speed': approx(102.9903, 0.001)},
        {'weekday': 'Fri', 'hour': 11, 'speed': approx(117.2945, 0.001)},
        {'weekday': 'Sat', 'hour': 13, 'speed': approx(158.1709, 0.001)},
        {'weekday': 'Sun', 'hour': 13, 'speed': approx(34.2408, 0.001)},
        {'weekday': 'Tue', 'hour': 6, 'speed': approx(165.0966, 0.001)},
        {'weekday': 'Tue', 'hour': 14, 'speed': approx(54.7402, 0.001)},
        {'weekday': 'Wed', 'hour': 14, 'speed': approx(140.3635, 0.001)}
    ]

    g = algorithms.build_yandex_maps_graph('travel_times', 'lengths')

    result = g.run(
        travel_times=islice(cycle(iter(times)), len(times) * 5000),
        lengths=iter(lengths)
    )

    assert sorted(result, key=lambda x: (x['weekday'], x['hour'])) == \
        sorted(etalon, key=lambda x: (x['weekday'], x['hour']))


def test_left_join():
    first_graph = Graph(source='first_table')
    second_graph = Graph(source='second_table')
    first_graph.join(on=second_graph, key='country_id', strategy='left')

    first_table = [
        {'country_id': 1, 'name': 'John', 'surname': 'Black'},
        {'country_id': 1, 'name': 'Antony', 'surname': 'Brown'},
        {'country_id': 2, 'name': 'Alex', 'surname': 'Sidorov'},
        {'country_id': 4, 'name': 'Frodo', 'surname': 'Ivanov'},
        {'country_id': 4, 'name': 'Bilbo', 'surname': 'Beggins'},
        {'country_id': 4, 'name': 'Frank', 'surname': 'Sinatra'},
        {'country_id': 6, 'name': 'Xiao', 'surname': 'Hao'},
    ]

    second_table = [
        {'country_id': 2, 'capital': 'Moscow'},
        {'country_id': 4, 'capital': 'Fairytail'},
        {'country_id': 5, 'capital': 'New York'},
    ]

    etalon_first = [
        {'country_id_left': 1, 'name': 'John', 'surname': 'Black', 'country_id_right': None, 'capital': None},
        {'country_id_left': 1, 'name': 'Antony', 'surname': 'Brown', 'country_id_right': None, 'capital': None},
        {'country_id_left': 2, 'name': 'Alex', 'surname': 'Sidorov', 'country_id_right': 2, 'capital': 'Moscow'},
        {'country_id_left': 4, 'name': 'Frodo', 'surname': 'Ivanov', 'country_id_right': 4, 'capital': 'Fairytail'},
        {'country_id_left': 4, 'name': 'Bilbo', 'surname': 'Beggins', 'country_id_right': 4, 'capital': 'Fairytail'},
        {'country_id_left': 4, 'name': 'Frank', 'surname': 'Sinatra', 'country_id_right': 4, 'capital': 'Fairytail'},
        {'country_id_left': 6, 'name': 'Xiao', 'surname': 'Hao', 'country_id_right': None, 'capital': None}
    ]

    result_first = first_graph.run(first_table=first_table, second_table=second_table)
    assert etalon_first == result_first

    third_graph = Graph(source='first_table')
    fourth_graph = Graph(source='second_table')
    fourth_graph.join(on=third_graph, key='country_id', strategy='left')

    etalon_second = [
        {'country_id_left': 2, 'capital': 'Moscow', 'country_id_right': 2, 'name': 'Alex', 'surname': 'Sidorov'},
        {'country_id_left': 4, 'capital': 'Fairytail', 'country_id_right': 4, 'name': 'Frodo', 'surname': 'Ivanov'},
        {'country_id_left': 4, 'capital': 'Fairytail', 'country_id_right': 4, 'name': 'Bilbo', 'surname': 'Beggins'},
        {'country_id_left': 4, 'capital': 'Fairytail', 'country_id_right': 4, 'name': 'Frank', 'surname': 'Sinatra'},
        {'country_id_left': 5, 'capital': 'New York', 'country_id_right': None, 'name': None, 'surname': None}
    ]

    result_second = fourth_graph.run(first_table=first_table, second_table=second_table)
    assert etalon_second == result_second

    fifth_graph = Graph(source='third_table')
    sixth_graph = Graph(source='forth_table')
    fifth_graph.join(on=sixth_graph, key=('id', 'user_id'), strategy='left')

    third_table = [
        {'id': 1, 'mail': 'nsa@yandex.ru'},
        {'id': 2, 'mail': 'sds@mail.ru'},
    ]

    forth_table = [
        {'user_id': 1, 'message': 'this is text'},
        {'user_id': 3, 'message': 'some text'},
        {'user_id': 1, 'message': 'hello'},
        {'user_id': 2, 'message': 'some text'},
        {'user_id': 1, 'message': 'lolololo'},
        {'user_id': 2, 'message': 'hi'},
        {'user_id': 2, 'message': 'python__'},
        {'user_id': 3, 'message': 'qeasd'},
        {'user_id': 2, 'message': 'kek'},
        {'user_id': 4, 'message': 'wew'},
        {'user_id': 2, 'message': 'pewpew'}
    ]

    etalon_third = [
        {'user_id': 1, 'message': 'this is text', 'id': 1, 'mail': 'nsa@yandex.ru'},
        {'user_id': 1, 'message': 'hello', 'id': 1, 'mail': 'nsa@yandex.ru'},
        {'user_id': 2, 'message': 'some text', 'id': 2, 'mail': 'sds@mail.ru'},
        {'user_id': 1, 'message': 'lolololo', 'id': 1, 'mail': 'nsa@yandex.ru'},
        {'user_id': 2, 'message': 'hi', 'id': 2, 'mail': 'sds@mail.ru'},
        {'user_id': 2, 'message': 'python__', 'id': 2, 'mail': 'sds@mail.ru'},
        {'user_id': 2, 'message': 'kek', 'id': 2, 'mail': 'sds@mail.ru'},
        {'user_id': 2, 'message': 'pewpew', 'id': 2, 'mail': 'sds@mail.ru'}
    ]
    result_third = fifth_graph.run(third_table=third_table, forth_table=forth_table)

    assert sorted(etalon_third, key=lambda x: (x['user_id'], x['message'])) == \
        sorted(result_third, key=lambda x: (x['user_id'], x['message']))


def test_inner_join():
    first_graph = Graph(source='first_table')
    second_graph = Graph(source='second_table')
    first_graph.join(on=second_graph, key='country_id', strategy='inner')

    first_table = [
        {'country_id': 1, 'name': 'John', 'surname': 'Black'},
        {'country_id': 1, 'name': 'Antony', 'surname': 'Brown'},
        {'country_id': 2, 'name': 'Alex', 'surname': 'Sidorov'},
        {'country_id': 4, 'name': 'Frodo', 'surname': 'Ivanov'},
        {'country_id': 4, 'name': 'Bilbo', 'surname': 'Beggins'},
        {'country_id': 4, 'name': 'Frank', 'surname': 'Sinatra'},
        {'country_id': 6, 'name': 'Xiao', 'surname': 'Hao'},
    ]

    second_table = [
        {'country_id': 2, 'capital': 'Moscow'},
        {'country_id': 4, 'capital': 'Fairytail'},
        {'country_id': 5, 'capital': 'New York'},
    ]

    etalon_first = [
        {'country_id_left': 2, 'name': 'Alex', 'surname': 'Sidorov', 'country_id_right': 2, 'capital': 'Moscow'},
        {'country_id_left': 4, 'name': 'Frodo', 'surname': 'Ivanov', 'country_id_right': 4, 'capital': 'Fairytail'},
        {'country_id_left': 4, 'name': 'Bilbo', 'surname': 'Beggins', 'country_id_right': 4, 'capital': 'Fairytail'},
        {'country_id_left': 4, 'name': 'Frank', 'surname': 'Sinatra', 'country_id_right': 4, 'capital': 'Fairytail'},
    ]

    result_first = first_graph.run(first_table=first_table, second_table=second_table)
    assert etalon_first == result_first

    fifth_graph = Graph(source='third_table')
    sixth_graph = Graph(source='forth_table')
    fifth_graph.join(on=sixth_graph, key=('id', 'user_id'), strategy='inner')

    third_table = [
        {'id': 1, 'mail': 'nsa@yandex.ru'},
        {'id': 2, 'mail': 'sds@mail.ru'},
    ]

    forth_table = [
        {'user_id': 1, 'message': 'this is text'},
        {'user_id': 3, 'message': 'some text'},
        {'user_id': 1, 'message': 'hello'},
        {'user_id': 2, 'message': 'some text'},
        {'user_id': 1, 'message': 'lolololo'},
        {'user_id': 2, 'message': 'hi'},
        {'user_id': 2, 'message': 'python__'},
        {'user_id': 3, 'message': 'qeasd'},
        {'user_id': 2, 'message': 'kek'},
        {'user_id': 4, 'message': 'wew'},
        {'user_id': 2, 'message': 'pewpew'}
    ]

    etalon_third = [
        {'user_id': 1, 'message': 'this is text', 'id': 1, 'mail': 'nsa@yandex.ru'},
        {'user_id': 1, 'message': 'hello', 'id': 1, 'mail': 'nsa@yandex.ru'},
        {'user_id': 2, 'message': 'some text', 'id': 2, 'mail': 'sds@mail.ru'},
        {'user_id': 1, 'message': 'lolololo', 'id': 1, 'mail': 'nsa@yandex.ru'},
        {'user_id': 2, 'message': 'hi', 'id': 2, 'mail': 'sds@mail.ru'},
        {'user_id': 2, 'message': 'python__', 'id': 2, 'mail': 'sds@mail.ru'},
        {'user_id': 2, 'message': 'kek', 'id': 2, 'mail': 'sds@mail.ru'},
        {'user_id': 2, 'message': 'pewpew', 'id': 2, 'mail': 'sds@mail.ru'}
    ]
    result_third = fifth_graph.run(third_table=third_table, forth_table=forth_table)

    assert sorted(etalon_third, key=lambda x: (x['user_id'], x['message'])) == \
        sorted(result_third, key=lambda x: (x['user_id'], x['message']))


def test_right_join():
    first_graph = Graph(source='first_table')
    second_graph = Graph(source='second_table')
    first_graph.join(on=second_graph, key='country_id', strategy='right')

    first_table = [
        {'country_id': 1, 'name': 'John', 'surname': 'Black'},
        {'country_id': 1, 'name': 'Antony', 'surname': 'Brown'},
        {'country_id': 2, 'name': 'Alex', 'surname': 'Sidorov'},
        {'country_id': 4, 'name': 'Frodo', 'surname': 'Ivanov'},
        {'country_id': 4, 'name': 'Bilbo', 'surname': 'Beggins'},
        {'country_id': 4, 'name': 'Frank', 'surname': 'Sinatra'},
        {'country_id': 6, 'name': 'Xiao', 'surname': 'Hao'},
    ]

    second_table = [
        {'country_id': 2, 'capital': 'Moscow'},
        {'country_id': 4, 'capital': 'Fairytail'},
        {'country_id': 5, 'capital': 'New York'},
    ]

    etalon_first = [
        {'country_id_left': 2, 'name': 'Alex', 'surname': 'Sidorov', 'country_id_right': 2, 'capital': 'Moscow'},
        {'country_id_left': 4, 'name': 'Frodo', 'surname': 'Ivanov', 'country_id_right': 4, 'capital': 'Fairytail'},
        {'country_id_left': 4, 'name': 'Bilbo', 'surname': 'Beggins', 'country_id_right': 4, 'capital': 'Fairytail'},
        {'country_id_left': 4, 'name': 'Frank', 'surname': 'Sinatra', 'country_id_right': 4, 'capital': 'Fairytail'},
        {'country_id_left': None, 'name': None, 'surname': None, 'country_id_right': 5, 'capital': 'New York'}
    ]

    result_first = first_graph.run(first_table=first_table, second_table=second_table)
    assert etalon_first == result_first


def test_full_join():
    first_graph = Graph(source='first_table')
    second_graph = Graph(source='second_table')
    first_graph.join(on=second_graph, key='country_id', strategy='full')

    first_table = [
        {'country_id': 1, 'name': 'John', 'surname': 'Black'},
        {'country_id': 1, 'name': 'Antony', 'surname': 'Brown'},
        {'country_id': 2, 'name': 'Alex', 'surname': 'Sidorov'},
        {'country_id': 4, 'name': 'Frodo', 'surname': 'Ivanov'},
        {'country_id': 4, 'name': 'Bilbo', 'surname': 'Beggins'},
        {'country_id': 4, 'name': 'Frank', 'surname': 'Sinatra'},
        {'country_id': 6, 'name': 'Xiao', 'surname': 'Hao'},
    ]

    second_table = [
        {'country_id': 2, 'capital': 'Moscow'},
        {'country_id': 4, 'capital': 'Fairytail'},
        {'country_id': 5, 'capital': 'New York'},
    ]

    etalon_first = [
        {'country_id_left': 1, 'name': 'John', 'surname': 'Black', 'country_id_right': None, 'capital': None},
        {'country_id_left': 1, 'name': 'Antony', 'surname': 'Brown', 'country_id_right': None, 'capital': None},
        {'country_id_left': 2, 'name': 'Alex', 'surname': 'Sidorov', 'country_id_right': 2, 'capital': 'Moscow'},
        {'country_id_left': 4, 'name': 'Frodo', 'surname': 'Ivanov', 'country_id_right': 4, 'capital': 'Fairytail'},
        {'country_id_left': 4, 'name': 'Bilbo', 'surname': 'Beggins', 'country_id_right': 4, 'capital': 'Fairytail'},
        {'country_id_left': 4, 'name': 'Frank', 'surname': 'Sinatra', 'country_id_right': 4, 'capital': 'Fairytail'},
        {'country_id_left': None, 'name': None, 'surname': None, 'country_id_right': 5, 'capital': 'New York'},
        {'country_id_left': 6, 'name': 'Xiao', 'surname': 'Hao', 'country_id_right': None, 'capital': None},
    ]

    result_first = first_graph.run(first_table=first_table, second_table=second_table)
    assert etalon_first == result_first


def test_cross_join():
    first_graph = Graph(source='first_table')
    second_graph = Graph(source='second_table')
    first_graph.join(on=second_graph, strategy='cross')

    first_table = [
        {'country_id': 1, 'name': 'John', 'surname': 'Black'},
        {'country_id': 1, 'name': 'Antony', 'surname': 'Brown'},
        {'country_id': 2, 'name': 'Alex', 'surname': 'Sidorov'},
        {'country_id': 6, 'name': 'Xiao', 'surname': 'Hao'},
    ]

    second_table = [
        {'capital': 'Moscow', 'description': 'description text'},
        {'capital': 'Fairytail', 'description': 'some text'},
    ]

    etalon_first = [
        {'country_id': 1, 'name': 'John', 'surname': 'Black', 'capital': 'Moscow', 'description': 'description text'},
        {'country_id': 1, 'name': 'John', 'surname': 'Black', 'capital': 'Fairytail', 'description': 'some text'},
        {'country_id': 1, 'name': 'Antony', 'surname': 'Brown', 'capital': 'Moscow', 'description': 'description text'},
        {'country_id': 1, 'name': 'Antony', 'surname': 'Brown', 'capital': 'Fairytail', 'description': 'some text'},
        {'country_id': 2, 'name': 'Alex', 'surname': 'Sidorov',
            'capital': 'Moscow', 'description': 'description text'},
        {'country_id': 2, 'name': 'Alex', 'surname': 'Sidorov', 'capital': 'Fairytail', 'description': 'some text'},
        {'country_id': 6, 'name': 'Xiao', 'surname': 'Hao', 'capital': 'Moscow', 'description': 'description text'},
        {'country_id': 6, 'name': 'Xiao', 'surname': 'Hao', 'capital': 'Fairytail', 'description': 'some text'}
    ]

    result_first = first_graph.run(first_table=first_table, second_table=second_table)
    assert sorted(etalon_first, key=lambda x: (x['country_id'], x['name'], x['capital'])) == \
        sorted(result_first, key=lambda x: (x['country_id'], x['name'], x['capital']))


def test_map():

    def mapper(record):
        record['value'] += 1
        yield record

    my_graph = Graph(source='table')

    table = [
        {'value': 1, 'text': 'some text'},
        {'value': 123, 'text': 'hello'},
        {'value': 55, 'text': 'hi'},
        {'value': 151, 'text': 'aaaAAa'}
    ]

    etalon = [
        {'value': 2, 'text': 'some text'},
        {'value': 124, 'text': 'hello'},
        {'value': 56, 'text': 'hi'},
        {'value': 152, 'text': 'aaaAAa'}
    ]
    my_graph.map(mapper)
    result = my_graph.run(table=table)
    assert result == etalon


def test_sort():

    table = [
        {'value': 1, 'text': 'some text'},
        {'value': 123, 'text': 'hello'},
        {'value': 55, 'text': 'week'},
        {'value': 55, 'text': 'hi'},
        {'value': 55, 'text': 'anananan'},
        {'value': 151, 'text': 'aaaAAa'}
    ]

    etalon = [
        {'value': 1, 'text': 'some text'},
        {'value': 55, 'text': 'week'},
        {'value': 55, 'text': 'hi'},
        {'value': 55, 'text': 'anananan'},
        {'value': 123, 'text': 'hello'},
        {'value': 151, 'text': 'aaaAAa'}
    ]

    my_graph = Graph(source='table')
    my_graph.sort('value')

    result = my_graph.run(table=table)
    assert result == etalon

    my_graph = Graph(source='table')
    my_graph.sort('value', 'text')

    second_etalon = [
        {'value': 1, 'text': 'some text'},
        {'value': 55, 'text': 'anananan'},
        {'value': 55, 'text': 'hi'},
        {'value': 55, 'text': 'week'},
        {'value': 123, 'text': 'hello'},
        {'value': 151, 'text': 'aaaAAa'}
    ]

    result = my_graph.run(table=table)
    assert result == second_etalon


def test_fold():

    def folder(state, record):
        for column in state:
            state[column] += record[column]
        return state

    table = [
        {'first_value': 1, 'second_value': 9.18},
        {'first_value': 123, 'second_value': 1999.1123},
        {'first_value': 28, 'second_value': 1.0},
        {'first_value': 55, 'second_value': 0.82821},
        {'first_value': 16, 'second_value': 1.1992},
        {'first_value': 151, 'second_value': 192.19002}
    ]

    etalon = [
        {'first_value': 374, 'second_value': approx(2203.50973, 0.001)}
    ]

    my_graph = Graph(source='table')
    my_graph.fold(folder, {'first_value': 0, 'second_value': 0.0})

    result = my_graph.run(table=table)
    assert result == etalon


def test_reduce():

    def reducer(records):
        word_count = Counter()

        for record in records:
            word_count[record['word']] += 1

        total = sum(word_count.values())
        for word, count in word_count.items():
            yield {
                'word': word,
                'freq': count / total,
                'group': record['group']
            }

    table = [
        {'word': 'animation', 'group': 'child'},
        {'word': 'binary', 'group': 'programming'},
        {'word': 'animation', 'group': 'child'},
        {'word': 'animation', 'group': 'child'},
        {'word': 'hi', 'group': 'child'},
        {'word': 'pyython', 'group': 'programming'}
    ]

    etalon = [
        {'word': 'animation', 'freq': approx(0.75, 0.001), 'group': 'child'},
        {'word': 'hi', 'freq': approx(0.25, 0.001), 'group': 'child'},
        {'word': 'binary', 'freq': approx(0.5, 0.001), 'group': 'programming'},
        {'word': 'pyython', 'freq': approx(0.5, 0.001), 'group': 'programming'}
    ]

    my_graph = Graph(source='table')
    my_graph.sort('group')
    my_graph.reduce(reducer, key='group')

    result = my_graph.run(table=table)
    assert result == etalon
