import json


class JobsData:

    jobs = []
    times = []
    positions = []

    @staticmethod
    def load_jobs():
        # read file
        with open('jobs.json', 'r') as jobs:
            data = jobs.read()

        # parse file
        obj = json.loads(data)

        # save values
        JobsData.jobs = obj['jobs']
        JobsData.times = obj['times']
        JobsData.positions = obj['positions']
