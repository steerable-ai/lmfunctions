import subprocess


def cuda_check():
    gpu_info = {"cuda_available": False, "num_gpus": 0, "gpus": []}

    try:
        # Check for CUDA availability using nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return gpu_info

        gpu_names = result.stdout.strip().split("\n")
        gpu_info["cuda_available"] = True
        gpu_info["num_gpus"] = len(gpu_names)

        # Query GPU memory
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        memory_info = result.stdout.strip().split("\n")

        for i, line in enumerate(memory_info):
            total_memory, free_memory, used_memory = line.split(", ")
            gpu_info["gpus"].append(
                {
                    "name": gpu_names[i],
                    "total_memory": int(total_memory),
                    "free_memory": int(free_memory),
                    "used_memory": int(used_memory),
                }
            )

    except FileNotFoundError:
        pass

    return gpu_info
