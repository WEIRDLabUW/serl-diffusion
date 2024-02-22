import subprocess

SCRIPT_NAME = "job.slurm"
GPUS = [4,5,6]
def run_one_job(gpu_id):
    env_command = f"CUDA_VISIBLE_DEVICES={gpu_id}"

    command = [
        "sbatch",
        "--export=ALL," + env_command,  # Export CUDA_VISIBLE_DEVICES to job
        SCRIPT_NAME

    ]

    # Submit the job
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output of the sbatch command (job submission result)
    print(result.stdout)


for i in range(10):
    gpu_id = GPUS[i % len(GPUS)]
    run_one_job(i)
