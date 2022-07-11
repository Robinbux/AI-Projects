import dimod
import neal
import time
import hybrid
from dwave_qbsolv import QBSolv
import dwave_qbsolv

from Flags.Flags import Flags

# Used for measuring time of sampling
def timing_decorator(func):
    if not Flags.verbose: return

    def wrapper(*args, **kwargs):
        start = time.time()
        original_return_val = func(*args, **kwargs)
        end = time.time()
        print("time elapsed in ", func.__name__, ": ", end - start, sep='')
        return original_return_val

    return wrapper


class Sampler:
    num_reads = 1000  # Default number of samples if not specified in the flags

    def sim_sample(self, bqm):
        sampler = neal.SimulatedAnnealingSampler()
        if Flags.num_reads is not None:
            self.num_reads = Flags.num_reads
        return sampler.sample(bqm=bqm, num_reads=self.num_reads)

    def qbsolv_solution(self,bqm):
        algo=dwave_qbsolv.ENERGY_IMPACT
        response = QBSolv().sample(bqm,verbosity=1,num_repeats=10000,solver_limit=1024,timeout=120,algorithm=algo)
        print(((response.data_vectors['energy']))[0])
        return response

    def real_sample(self, bqm):
        """
        For now using the Kerberos Hybrid Sampler, as otherwise the possible
        problem size is too little
        """
        
        return hybrid.KerberosSampler().sample(bqm)

    def sample(self, QUBO):
        bqm = dimod.BQM.from_qubo(QUBO)
        if Flags.quantum:
            return self.real_sample(bqm)
        # result = self.sim_sample(bqm)
        result = self.qbsolv_solution(bqm)
        return result
