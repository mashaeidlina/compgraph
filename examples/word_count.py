from algorithms import build_word_count_graph


def main() -> None:
    """
    Calls function build_word_count_graph() from tests/algorithms.py with data from resources/text_corpus.txt
    and saves result to results/word_count.txt
    :return: returns nothing
    """
    with open("resources/text_corpus.txt", 'r') as input_file:
        my_graph = build_word_count_graph('input_stream', count_column='count', text_column='text')
        with open("results/word_count.txt", 'w') as output_file:
            my_graph.run(input_stream=input_file, output_stream=output_file)


if __name__ == "__main__":
    main()
