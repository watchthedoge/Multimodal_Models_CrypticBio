import argparse
import sys


def run_date():
    from date_experiment import run
    run()


def run_location():
    from location_experiment import run
    run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='experiment',
        description='What experiment is chosen'
    )
    parser.add_argument(
        '--e',
        nargs='?',
        choices=['location', 'date', 'both','sinr','sinr_ft'],
        required=True,
        help='choose from "location", "date", sinr. "sinr_ft" or "both"'
    )

    args = parser.parse_args()

    if args.e == "date":
        run_date()
    elif args.e == "location":
        run_location()
    elif args.e == "both":
        run_date()
        run_location()
    elif args.e == "sinr":
        import sinr_location_experiment
        sinr_location_experiment.run_sweep()
