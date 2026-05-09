import argparse


def run_date(benchmark):
    from date_experiment import run
    run(benchmark=benchmark)


def run_location(benchmark):
    from location_experiment import run
    run(benchmark=benchmark)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='experiment',
        description='What experiment is chosen'
    )
    parser.add_argument(
        '--e',
        nargs='?',
        choices=['location', 'date', 'both', 'sinr', 'sinr_ft'],
        required=True,
        help='choose from "location", "date", sinr. "sinr_ft" or "both"'
    )
    parser.add_argument(
        '--benchmark',
        nargs='?',
        choices=['common', 'common_unseen', 'endangered'],
        default='common',
        help='which CrypticBio subset to train+evaluate on (default: common)'
    )

    args = parser.parse_args()

    if args.e == "date":
        run_date(args.benchmark)
    elif args.e == "location":
        run_location(args.benchmark)
    elif args.e == "both":
        run_date(args.benchmark)
        run_location(args.benchmark)
    elif args.e == "sinr":
        import sinr_location_experiment
        sinr_location_experiment.run_sweep(benchmark=args.benchmark)