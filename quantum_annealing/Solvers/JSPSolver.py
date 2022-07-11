from Flags.Flags import Flags
from JobsLoader.JobsData import JobsData
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt


class JSPSolver:
    QUBO = np.zeros((0, 0))
    qubo_initialized = False

    def __init__(self):
        self.NUMBER_OF_OPERATIONS = self.__get_number_of_operations()
        self.NUMBER_OF_MACHINES = self.__get_number_of_machines()
        self.NUMBER_OF_BOTS = len(
            JobsData.positions)  # Temporarily as I did not account for the positions in the initial version
        self.NUMBER_OF_WALKING_OPERATIONS = self.__get_number_of_walking_operations()
        self.NUMBER_OF_INDIVIDUAL_WALKING_OPERATIONS = self.NUMBER_OF_WALKING_OPERATIONS * self.NUMBER_OF_BOTS
        self.MAX_WALK_TIME = self.__get_max_walk_time()
        self.FLATTENED_OPERATIONS = self.__merge_operations()
        if Flags.time_limit is not None:
            self.UPPER_TIME_LIMIT = Flags.time_limit
        else:
            self.UPPER_TIME_LIMIT = self.__get_worst_upper_time_limit_range()
        if not JSPSolver.qubo_initialized:
            self.BASEUPPERBOUNDVARS=self.UPPER_TIME_LIMIT * (
                    self.NUMBER_OF_OPERATIONS + self.NUMBER_OF_INDIVIDUAL_WALKING_OPERATIONS)
            QUBO_LENGTH = self.BASEUPPERBOUNDVARS+ self.UPPER_TIME_LIMIT
            JSPSolver.QUBO = np.zeros((QUBO_LENGTH, QUBO_LENGTH))
            JSPSolver.qubo_initialized = True


    def get_operation_x(self, x):
        return self.FLATTENED_OPERATIONS[x]

    def get_operation_indexes_for_machine_m(self, m):
        indexes = []
        op_idx = 0
        for j in JobsData.jobs:
            for o in j:
                if o[0] == m:
                    indexes.append(op_idx)
                op_idx += 1
        return indexes

    def get_operation_indexes_for_job_j(self, j):
        op_idx = 0
        for i in range(j):
            op_idx += len(JobsData.jobs[i])
        return list(range(op_idx, op_idx + len(JobsData.jobs[j])))

    def fill_QUBO_with_indexes(self, i, t, i_prime, t_prime, value):
        index_a = self.convert_two_indices_to_one(i, t)
        index_b = self.convert_two_indices_to_one(i_prime, t_prime)
        if index_a > index_b:
            index_a, index_b = index_b, index_a
        JSPSolver.QUBO[index_a][index_b] += value

    def response_quick_check_must_fix_later(self, response):
        operation_results = {}
        for k, v in response.first[0].items():
            if v == 1:
                res = self.convert_one_index_to_two(k)
                if res[0] in operation_results:
                    if res[0] >= self.NUMBER_OF_OPERATIONS:  # TODO: Find a more elegant solution, but I'm tired now
                        return 'h4'
                    return 'h1'
                operation_results[res[0]] = res[1]
        return None

    def convert_response_to_operation_results(self, response):
        operation_results = {}
        for k, v in response.first[0].items():
            if v == 1:
                if k<self.BASEUPPERBOUNDVARS:
                    res = self.convert_one_index_to_two(k)
                    operation_results[res[0]] = res[1]
                else:
                    print("upperbound: ",k-self.BASEUPPERBOUNDVARS)                    
        return operation_results

    def convert_response(self, response):
        operation_results = {}
        for k, v in response.first[0].items():
            if v == 1:
                res = self.convert_one_index_to_two(k)
                if k<self.BASEUPPERBOUNDVARS:
                     operation_results[res[0]] = res[1]
                else:
                    print("upperbound: ",k-self.BASEUPPERBOUNDVARS)
        return operation_results

    def operation_results_to_plot_format(self, operation_results):
        result = {}
        result["schedule"] = JobsData.jobs
        result["times"] = JobsData.times
        result["transport"] = []

        # Extend standard operations
        current_op = 0
        for j in range(len(JobsData.jobs)):
            for i in range(len(JobsData.jobs[j])):
                result["schedule"][j][i].append(operation_results[current_op])
                current_op += 1

        # Bot operations
        for b in range(self.NUMBER_OF_BOTS):
            bot_ops = []
            for i in self.get_walking_operation_indexes_for_bot_b(b):
                if i in operation_results:
                    walking_op = self.get_operation_x(i)
                    job_nbr = self.__get_job_for_operation_x(i)
                    bot_ops.append([walking_op[2], walking_op[3], operation_results[i], job_nbr])
            result["transport"].append(bot_ops)
        return result

    def __get_job_for_operation_x(self, x):
        if x < self.NUMBER_OF_OPERATIONS:
            return -1
        x -= self.NUMBER_OF_OPERATIONS
        passed_walking_ops = 0
        for idx, job in enumerate(JobsData.jobs):
            walking_ops_in_current_job = (len(job) - 1) * self.NUMBER_OF_BOTS
            passed_walking_ops += walking_ops_in_current_job
            if x < passed_walking_ops:
                return idx

    cols = 'rbkgymc'
    syms = ['+', 'o', '*']

    def bar2(self, time, duration, machine, job, task, nrjobs, agvnr, nrmachines, nragvs):
        plt.plot([time, time + duration, time + duration, time, time],
                 machine + (np.array([0, 0, 1, 1, 0]) - 0.5) * 0.5, self.cols[job])
        plt.text(time + duration * 0.5, machine + 0.3, str(duration))
        plt.text(time + 0.1, machine, str(job) + "." + str(task))
        plt.text(time, machine - 0.4, str(time))
        plt.plot([time + duration, time + duration], [-1, 1 + nrmachines + nragvs], ':', color='gray')

    # plots a standard json solution
    def printstandardsol(self, solution, name):
        j = solution['schedule']
        nrjobs = len(j)
        transportops = solution['transport']
        t = solution['times']
        nragvs = len(transportops)
        nrmachines = 1 + np.max([machine for job in j for machine, duration, start in job])
        mx = 0
        for job in range(nrjobs):
            for task in range(len(j[job])):
                oper = j[job][task]
                time = oper[2]
                machine = oper[0]
                duration = oper[1]
                self.bar2(time, duration, machine, job, task, nrjobs, -1, nrmachines, nragvs)
                end = time + duration
                if end > mx:
                    mx = end
        for agvnr in range(len(transportops)):
            transportops1 = transportops[agvnr]
            for fromm, tomm, time, job in transportops1:
                duration = t[fromm][tomm]
                ofs = agvnr / 20
                if job >= 0:
                    style = self.cols[job] + '-'
                else:
                    style = "k:"
                plt.plot([time, time + duration], [ofs + fromm, ofs + tomm], style + self.syms[agvnr], markersize=16)
                self.bar2(time, duration, nrmachines + agvnr, job, -1, nrjobs, -1, nrmachines, nragvs)
        plt.xticks(np.arange(0, mx + 1, 1.0))
        plt.grid(b=True, which='both', ls='-', color='gray', alpha=0.5)
        plt.savefig(name)
        plt.show()

        print("Effective horizon =", mx)

    def plot_operations(self, response):
        if not Flags.plot_graph:
            return
        operation_results = self.convert_response_to_operation_results(response)
        result_plot_format = self.operation_results_to_plot_format(operation_results)
        self.printstandardsol(result_plot_format, "Solution")


    def get_operation_indexes_for_walking_operation_w(self, w):
        """Return a list off all indexes of Ww"""
        start_point = self.NUMBER_OF_OPERATIONS + w * self.NUMBER_OF_BOTS
        return list(range(start_point, start_point + self.NUMBER_OF_BOTS))

    def get_walking_operation_indexes_for_bot_b(self, b):
        """Return a list of all walking operation indexes for Bot b"""
        return np.arange(self.NUMBER_OF_OPERATIONS + b, len(self.FLATTENED_OPERATIONS), self.NUMBER_OF_BOTS).tolist()

    def get_walking_operations_indexes_for_standard_operation_x(self, x):
        current_big_w_passed = 0
        current_op_passed = 0
        for j in JobsData.jobs:
            if current_op_passed + len(j) <= x:
                current_op_passed += len(j)
                current_big_w_passed += len(j) - 1
            else:
                w = x - current_op_passed + current_big_w_passed - 1
                if w < current_big_w_passed:
                    return []
                return self.get_operation_indexes_for_walking_operation_w(w)

    def get_standard_operation_index_before_individual_walking_operation_w(self, w):
        w = self.__transform_individual_to_big_walking_operation(w)
        current_big_w_passed = 0
        for idx, j in enumerate(JobsData.jobs):
            if current_big_w_passed + len(j) - 1 <= w:
                current_big_w_passed += len(j) - 1
            else:
                return w + idx

    def print_operation_results(self, response):
        operation_results = self.convert_response_to_operation_results(response)
        op_count = 0
        print("Standard Operations")
        print("----------------------------------")
        for (k, v) in operation_results.items():
            if op_count == self.NUMBER_OF_OPERATIONS:
                print("\nBot Operations")
                print("----------------------------------")
            print(f"Operation {str(k)} starts at time {str(v)}")
            op_count += 1

    def plot_matrix(self):
        if not Flags.show_matrix:
            return
        x = []
        y = []
        for o in range(len(self.FLATTENED_OPERATIONS)):
            for t in range(self.UPPER_TIME_LIMIT):
                x.append("X " + str(o) + "," + str(t))
                y.append("X " + str(o) + "," + str(t))

        y = list(reversed(y))

        fig = ff.create_annotated_heatmap(z=np.flip(self.QUBO, 0).astype(int), x=x, y=y)
        fig.show(renderer="browser")

    def __get_worst_upper_time_limit_range(self):
        """Not quite accurate with the bots, but best I have for now"""
        max_time = 0
        for job in JobsData.jobs:
            for operation in job:
                max_time += operation[1]
        return max_time

    def __transform_individual_to_big_walking_operation(self, w):
        return int((w - self.NUMBER_OF_OPERATIONS) / self.NUMBER_OF_BOTS)

    def __merge_operations(self):
        """Flatten out operations for easier access and add bot operations at the end"""
        merged_operations = []
        # Flatten the operations
        for j in JobsData.jobs:
            merged_operations += j
        # Add the bot operations
        for j in JobsData.jobs:
            for o in range(len(j) - 1):
                for b in range(self.NUMBER_OF_BOTS):
                    start_machine = j[o][0]
                    end_machine = j[o + 1][0]
                    walking_time = JobsData.times[start_machine][end_machine]
                    merged_operations += [(b, walking_time, start_machine, end_machine)]
        return merged_operations

    def __get_number_of_operations(self):
        nbr_operations = 0
        for j in JobsData.jobs:
            nbr_operations += len(j)
        return nbr_operations

    def __get_number_of_walking_operations(self):
        nbr_bot_operations = 0
        for j in JobsData.jobs:
            nbr_bot_operations += len(j) - 1
        return nbr_bot_operations

    def __get_number_of_machines(self):
        machines = set()
        for j in JobsData.jobs:
            for o in j:
                machines.add(o[0])
        return len(machines)

    def __get_max_walk_time(self):
        max_walk_time = 0
        for w in JobsData.times:
            if max(w) > max_walk_time:
                max_walk_time = max(w)
        return max_walk_time

    def convert_two_indices_to_one(self, i, j):
        return i * self.UPPER_TIME_LIMIT + j

    def convert_one_index_to_two(self, k):
        j = k % self.UPPER_TIME_LIMIT
        i = int((k - j) / self.UPPER_TIME_LIMIT)
        return [i, j]

    def not_used(self):
        pass
