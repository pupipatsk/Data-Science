import os
import multiprocessing

def find_max_jobs(verbose=True) -> int:
    """
    Determine recommended number of jobs for parallel processing and optionally print the details.

    Calculates optimal job count considering:
    - Physical CPU cores
    - Logical CPU cores (including hyper-threading)

    Args:
        max_recommended_multiplier (int, optional): Multiplier for physical cores.
            Defaults to 2 to account for hyperthreading.
        verbose (bool, optional): If True, prints a summary of job information. Defaults to True.

    Returns:
        Dict containing system job information with keys:
        - 'physical_cores': Number of physical CPU cores
        - 'logical_cores': Total logical CPU cores
    """
    # Physical CPU cores
    physical_cores = os.cpu_count() or 1

    # Logical CPU cores
    logical_cores = max(multiprocessing.cpu_count(), 1)

    # Prepare job info dictionary
    job_info = {
        'physical_cores': physical_cores,
        'logical_cores': logical_cores,
    }

    # Print job info if verbose is True
    if verbose:
        print("CPU Job Information.")
        print(f"Physical CPU Cores: {job_info['physical_cores']}")
        print(f"Logical CPU Cores: {job_info['logical_cores']}")

    return max(physical_cores, logical_cores)

def main():
    """
    Main entry point for the script.
    """
    find_max_jobs()

if __name__ == "__main__":
    main()
