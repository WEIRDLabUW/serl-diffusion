import time

import submitit


def add(a, b):
    time.sleep(4)
    return a + b


def main():
    print("creating executor")
    jobs = []
    executor = submitit.AutoExecutor(folder="log_test")
    # set timeout in min, and partition for running the job
    executor.update_parameters(slurm_partition="gpu-a40",
                               slurm_account="weirdlab",
                               slurm_name="experiment",
                               timeout_min=1,
                               mem_gb=10,
                               slurm_gpus_per_node=1,
                               slurm_gpus_per_task=1,
                               slurm_ntasks_per_node=1,
                               )
    executor.update_parameters(slurm_array_parallelism=2)
    with executor.batch():
        # In here submit jobs, and add them to the list, but they are all considered to be batched.
        for i in range(8):
            job = executor.submit(add, 1, i)
            jobs.append(job)

    while jobs:
        done = []
        for i in range(len(jobs)):
            job = jobs[i]
            if job.done():
                done.append(i)
                print(job.result())
        for i in reversed(done):
            del jobs[i]


if __name__ == "__main__":
    main()
