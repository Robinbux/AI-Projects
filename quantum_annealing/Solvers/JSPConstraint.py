from JobsLoader.JobsData import JobsData
from Lagrange.Lagrange import Lagrange
from Solvers.JSPSolver import JSPSolver


class JSPConstraint(JSPSolver):

    def __init__(self):
        super().__init__()

    def add_constraints(self):
        self.__add_h1_constraint()
        self.__add_h2_constraint()
        self.__add_h3_constraint()
        self.__add_h8_constraint()
        self.__add_h9_constraint()        
        self.__add_objective_function()
    #
    # h1 implementation
    #
    def __add_h1_constraint(self):
        for i in range(self.NUMBER_OF_OPERATIONS):
            for u in range(self.UPPER_TIME_LIMIT):
                for t in range(u):
                    self.fill_QUBO_with_indexes(i, t, i, u, Lagrange.alpha * 2)
            for t in range(self.UPPER_TIME_LIMIT):
                self.fill_QUBO_with_indexes(i, t, i, t, -Lagrange.alpha)

    #
    # h2 implementation
    #
    def __add_h2_constraint(self):
        def Rm_condition_fulfilled(i, t, k, t_prime, M):
            return i != k and 0 <= t and t_prime <= M and 0 <= t_prime - t < self.get_operation_x(i)[1]

        for m in range(self.NUMBER_OF_MACHINES):
            operation_indexes_m = self.get_operation_indexes_for_machine_m(m)
            for i in operation_indexes_m:
                for i_prime in operation_indexes_m:
                    for t in range(self.UPPER_TIME_LIMIT):
                        for t_prime in range(self.UPPER_TIME_LIMIT):
                            if Rm_condition_fulfilled(i, t, i_prime, t_prime, self.UPPER_TIME_LIMIT):
                                self.fill_QUBO_with_indexes(i, t, i_prime, t_prime, Lagrange.beta)

    #
    # h3 implementation
    #
    def __add_h3_constraint(self):
        nbr_jobs = len(JobsData.jobs)
        for j in range(nbr_jobs):
            for i in self.get_operation_indexes_for_job_j(j)[:-1]:
                for t in range(self.UPPER_TIME_LIMIT):
                    for t_prime in range(self.UPPER_TIME_LIMIT):
                        if (t + self.get_operation_x(i)[1]) > t_prime:
                            self.fill_QUBO_with_indexes(i, t, i + 1, t_prime, Lagrange.gamma)

#exactly one uppertime is set
    def __add_h8_constraint(self):
        for t1 in range(self.UPPER_TIME_LIMIT):
            for t2 in range(self.UPPER_TIME_LIMIT):
                i1=self.BASEUPPERBOUNDVARS+t1
                i2=self.BASEUPPERBOUNDVARS+t2
                if t1<t2:
                   self.QUBO[i1][i2] += 2*Lagrange.alpha
                else: 
                    if t1==t2:
                        self.QUBO[i1][i2] -= Lagrange.alpha

#no operation/transportation ends after the set uppertime
    def __add_h9_constraint(self):
        for t1 in range(self.UPPER_TIME_LIMIT):
            i1=self.BASEUPPERBOUNDVARS+t1
            for t2 in range(self.UPPER_TIME_LIMIT):
                for op in range(self.NUMBER_OF_OPERATIONS):
                    i2 = self.convert_two_indices_to_one(op, t2)
                    if t2+self.get_operation_x(op)[1]>t1:
                        self.QUBO[i2][i1] += Lagrange.alpha

# the objective function: minimize upperbound
    def __add_objective_function(self):
        for t in range(self.UPPER_TIME_LIMIT):
            i=self.BASEUPPERBOUNDVARS+t
            self.QUBO[i][i] += t
