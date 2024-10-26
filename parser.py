def parse_command_line_arguments(argv : list[str]):
    if len(argv) != 3:
        raise Exception("Wrong usage: conditional_hill_estimator.py [time_series_file] [kernel] [...]")

    #time_series = parse_excel_file(argv[1])
    #kernel = argv[2]

