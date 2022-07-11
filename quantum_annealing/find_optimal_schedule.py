from Flags.Flags import Flags
from JobsLoader.JobsData import JobsData
from Lagrange.Lagrange import Lagrange
from Sampler.Sampler import Sampler
from Solvers.JSPConstraint import JSPConstraint
from Solvers.JSPSolver import JSPSolver
from Solvers.TurtlebotConstraint import TurtlebotConstraint
from utils.utils import print_response


def main(args=None):
    # Setup
    Lagrange.load_params()
    Flags.parse_arguments(args)
    JobsData.load_jobs()

    print("Starting...")

    jsp_constraint = JSPConstraint()
    jsp_constraint.add_constraints()

    turtlebot_constraint = TurtlebotConstraint()
    turtlebot_constraint.add_constraints()

    sampler = Sampler()

    response = sampler.sample(JSPSolver.QUBO)

    print_response(response)
    jsp_constraint.print_operation_results(response)
    jsp_constraint.plot_operations(response)
    jsp_constraint.plot_matrix()

    
if __name__ == "__main__":
    main()
