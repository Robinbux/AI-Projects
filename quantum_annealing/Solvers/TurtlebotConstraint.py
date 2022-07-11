from JobsLoader.JobsData import JobsData
from Lagrange.Lagrange import Lagrange
from Solvers.JSPSolver import JSPSolver


class TurtlebotConstraint(JSPSolver):

    def __init__(self):
        super().__init__()

    def add_constraints(self):
        self.__add_h4_constraint()
        self.__add_h5_constraint()
        self.__add_h6_constraint()
        self.__add_h7_constraint()

    #
    # h4 implementation
    #
    def __add_h4_constraint(self):
        for w in range(self.NUMBER_OF_WALKING_OPERATIONS):
            walking_operation_indexes = self.get_operation_indexes_for_walking_operation_w(w)
            for i in walking_operation_indexes:
                for t in range(self.UPPER_TIME_LIMIT):
                    self.fill_QUBO_with_indexes(i, t, i, t, Lagrange.delta)
                    for t_prime in range(self.UPPER_TIME_LIMIT):
                        if t != t_prime:
                            self.fill_QUBO_with_indexes(i, t, i, t_prime, Lagrange.delta)
                for i_prime in walking_operation_indexes:
                    if i_prime != i:
                        for t in range(self.UPPER_TIME_LIMIT):
                            for t_prime in range(self.UPPER_TIME_LIMIT):
                                self.fill_QUBO_with_indexes(i, t, i_prime, t_prime, Lagrange.delta)
                for t in range(self.UPPER_TIME_LIMIT):
                    self.fill_QUBO_with_indexes(i, t, i, t, - 2*Lagrange.delta)

    #
    # h5 implementation
    #
    def __add_h5_constraint(self):
        for b in range(self.NUMBER_OF_BOTS):
            walking_operation_indexes = self.get_walking_operation_indexes_for_bot_b(b)
            for i in walking_operation_indexes:
                for i_prime in walking_operation_indexes:
                    if i != i_prime:
                        for t_prime in range(self.UPPER_TIME_LIMIT):
                            for t in range(t_prime + 1):
                                machine_end_i = self.get_operation_x(i)[3]
                                machine_start_i_prime = self.get_operation_x(i_prime)[2]
                                walktime_between_machines = JobsData.times[machine_end_i][machine_start_i_prime]
                                penalty = max(t + self.get_operation_x(i)[1] + walktime_between_machines - t_prime, 0)
                                self.fill_QUBO_with_indexes(i, t, i_prime, t_prime, penalty * Lagrange.epsilon)

    #
    # h6 implementation
    #
    def __add_h6_constraint(self):
        for i in range(self.NUMBER_OF_OPERATIONS):
            for i_prime in self.get_walking_operations_indexes_for_standard_operation_x(i):
                for t in range(self.UPPER_TIME_LIMIT):
                    for t_prime in range(self.UPPER_TIME_LIMIT):
                        penalty = max(t_prime + self.get_operation_x(i_prime)[1] - t, 0)
                        self.fill_QUBO_with_indexes(i, t, i_prime, t_prime, penalty * Lagrange.zeta)

    #
    # h7 implementation
    #
    def __add_h7_constraint(self):
        for w in range(self.NUMBER_OF_WALKING_OPERATIONS):
            for i in self.get_operation_indexes_for_walking_operation_w(w):
                for t in range(self.UPPER_TIME_LIMIT):
                    for t_prime in range(self.UPPER_TIME_LIMIT):
                        standard_op_idx = self.get_standard_operation_index_before_individual_walking_operation_w(i)
                        penalty = max(t_prime + self.get_operation_x(standard_op_idx)[1] - t, 0)
                        self.fill_QUBO_with_indexes(i, t, standard_op_idx, t_prime, penalty * Lagrange.eta)