#ComputeGraph

This library is designed for easy calculation above tables.

A table is a sequence of dict-like objects (list of dictionaries or json file with strings specified as dictionaries).
Each dictionary is a table row while a dictionary key is a table column.

There is an example of table 'users' below:
```(Python)
users = [
    '{'country_id': 1, 'name': 'John', 'surname': 'Black'},
     {'country_id': 1, 'name': 'Antony', 'surname': 'Brown'},
     {'country_id': 2, 'name': 'Alex', 'surname': 'Sidorov'},
     {'country_id': 4, 'name': 'Frodo', 'surname': 'Ivanov'},
     {'country_id': 4, 'name': 'Bilbo', 'surname': 'Beggins'},
     {'country_id': 4, 'name': 'Frank', 'surname': 'Sinatra'},
     {'country_id': 6, 'name': 'Xiao', 'surname': 'Hao'},'
]
```

This library allows to set calculations above the tables using Computational Graphs and then to run them. 
So, the library allows you to define a chain of transformations on the tables so that these transformations
are performed separately from their definition.
Computational graph describes multi-stage processing of several tables, and can be applied for different tables.

###Computational graph interface

The Computational Graphs consists of inputs and operations on them.
When you set calculation it does not run. To run a graph, you should call a 'run' method of the Graph class.

As an input, the graph can take:
1) Iterable sequence (list) of python dicts or
2) Opened file with json-strings in dict format
3) Generator which produces dicts

There is an example of interface:

```python
def reducer(records):
        ...

table = [
    {'word': 'animation', 'group': 'child'},
    ...
]

my_graph = Graph(source='table') # Sets input
my_graph.sort('group') # Sets operations
my_graph.reduce(reducer, key='group') # Sets operations

result = my_graph.run(table=table) # Runs calculations
```

###Operations
There are 5 possible operations with tables:
####1. Map
Map is an operation that calls the passed generator (called mapper) from each of the table rows.
The dicts given by the generator form a result table. 

There is a simple example of mapper function (that adds 1 to all fields in column 'value'):

```python
def mapper(record):
    record['value'] += 1
    yield record
```

####2. Sort
Sort is an operation thar sorts table by some set of columns lexicographically.

####3. Fold
Fold "folds" a table into a single row using a binary associative operation.
The are two required arguments:
1) the folder function (generator) and
2) the initial state.

Folder function is called with two arguments: 1) current state and 2) record. Folder
returns new state, changed by the record.

Example (folder calculates sum of some columns of table):
```python
def sum_columns_folder(state, record):
    for column in state:
        state[column] += record[column]
    return state
```

####4. Reduce

Reduce is an operation similar to map, but called not for one row of the table, but for all rows with the same key value.
For the efficient operation, the table supplied to reducer input must be sorted by the columns on which it is run.

There is an example of reducer that calculates frequency of every word in document:
```python
def reducer(records):
    word_count = Counter()

    for record in records:
        word_count[record['word']] += 1

    total = sum(word_count.values())
    for word, count in word_count.items():
        yield {
            'word': word,
            'freq': count / total,
            'doc_id': record['doc_id']
        }
```

####5. Join
Join combines information from two tables and return one table.
The rows of the new table will be created from the rows of the two tables involved in the join.
There are five join strategies: 1) inner, 2) left, 3) right, 4) full and 5) cross.
Join takes another graph object as an argument.
If two tables have common keys (keys with the same name) in result table (join table) there will be new names for such keys:
name of keys for left table will have addition '_left', name of keys for right tabke will have addition '_right'.
```python
first_table = [
    {'id': 1, 'mail': 'nsa@yandex.ru'},
    {'id': 2, 'mail': 'sds@mail.ru'},
    ...
]

second_table = [
    {'user_id': 1, 'message': 'this is text'},
    {'user_id': 2, 'message': 'some text'},
    ...
]

first_graph = Graph(source='first_table')
second_graph = Graph(source='second_table')
first_graph.join(on=second_graph, key=('id', 'user_id'), strategy='inner')
    
result_third = first_graph.run(first_table=first_table, second_table=second_table)
```

The result_third could look like:
```python
result_third = [
    {'user_id': 1, 'message': 'this is text', 'id': 1, 'mail': 'nsa@yandex.ru'},
    {'user_id': 2, 'message': 'some text', 'id': 2, 'mail': 'sds@mail.ru'},
    ...
]
```

Using the implemented library 4 tasks were solved:
1) Word Count
2) An inverted index with tf-idf
3) Top words with the most mutual information
4) Average speed in the city from the hour and day of the week

###Setup
To setup the library you should go to folder with **setup.py** file.
Type the following command in bash:
```bash
pip3 install .
```

###Files
- **setup.py** - setup file
- folder 'tests' consists unit-tests for library and tests for 4 solved tasks (**tests/test_graph.py**)
The tasks' solution is in tests/algorithms.py
- the main realization of graph is in **graph/src/graph.py**
- examples of use are in **examples** (the data for examples in **examples/resources/**)




