from algorithms import build_yandex_maps_graph
import json
# import pandas as pd   # For plots


def main() -> None:
    """
    Calls function build_yandex_maps_graph() with data from resources/travel_times.txt and
    resources/graph_data.txt and saves result to results/yandex_maps.txt
    :return: returns nothing
    """
    with open("resources/travel_times.txt", 'r') as input_file_times:
        with open("resources/graph_data.txt") as input_file_coords:
            my_graph = build_yandex_maps_graph('input_stream_times', 'input_file_coords')
            result = my_graph.run(input_stream_times=input_file_times, input_file_coords=input_file_coords)

    with open("results/yandex_maps.txt", 'w') as output_file:
        for record in result:
            output_file.write(json.dumps(record) + "\n")

    """
    Was used for building plots (in examples/results)
    weeks_dict = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    df = pd.DataFrame(result)
    for weekday in weeks_dict.values():
        part_df = df[df['weekday'] == weekday]
        if not part_df.empty:
            my_plot = part_df.plot(x='hour', y='speed', title=weekday)
            fig = my_plot.get_figure()
            fig.savefig("results/yandex_maps_{}.pdf".format(weekday))
    """


if __name__ == "__main__":
    main()
