from algorithms import build_pmi_graph


def main() -> None:
    """
    Calls function build_inverted_index_graph() from tests/algorithms.py with data from resources/text_corpus.txt
    and saves result to results/pmi.txt
    :return: returns nothing
    """
    with open("resources/text_corpus.txt", 'r') as input_file:
        my_graph = build_pmi_graph('input_stream', doc_column='doc_id', text_column='text')
        with open("results/pmi.txt", 'w') as output_file:
            my_graph.run(input_stream=input_file, output_stream=output_file)


if __name__ == "__main__":
    main()
