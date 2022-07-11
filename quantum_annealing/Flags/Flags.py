import argparse


class Flags:
    verbose = False
    show_matrix = False
    plot_graph = False
    simulated = True
    quantum = False
    num_reads = None
    time_limit = None

    def parse_arguments(args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', "--verbose", dest='v', help="More verbose output.", action='store_true')
        parser.add_argument('-m', "--matrix", dest='m', help="Show an interactive confusion matrix of the final Q.",
                            action='store_true')
        parser.add_argument('-s', "--simulated", dest='s', help="Use the simulated annealer", default=True, action='store_true')
        parser.add_argument('-q', "--quantum", dest='q', help="Use the D-Wave quantum computer (Kerberos Hyrid)",
                            default=False, action='store_true')
        parser.add_argument('-p', "--plot", dest='p', help="Plot the graph", action='store_true')
        parser.add_argument(
            '-r', '--num-reads',
            dest='r',
            help='Number of reads performed',
            type=int,
            nargs='?',
            default=None,
            metavar='NUMBER_OF_READS')
        parser.add_argument(
            '-t', '--time-limit',
            dest='t',
            help='Estimated upper time limit',
            type=int,
            nargs='?',
            default=None,
            metavar='TIME_LIMIT')

        options, unknown = parser.parse_known_args(args=args) if args is not None else parser.parse_known_args()

        Flags.verbose = options.v
        Flags.show_matrix = options.m
        Flags.plot_graph = options.p
        Flags.simulated = options.s
        Flags.quantum = options.q
        Flags.num_reads = options.r
        Flags.time_limit = options.t

    @staticmethod
    def verbose_print(string):
        """Print the string ony in verbose mode"""
        if Flags.verbose:
            print(string)
