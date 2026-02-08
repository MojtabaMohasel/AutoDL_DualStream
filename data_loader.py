import glob
import os
import pickle
import re
import numpy as np

def load_individual_generations(checkpoints_dir, prefix):
    """
    Robustly loads all individual generation checkpoint files (*-gen<N>_*.pkl)
    from the archive directory, sorts them numerically, and returns them as a list.

    This method works even if the GA run was interrupted.
    """
    gen_files = glob.glob(os.path.join(checkpoints_dir, f"{prefix}-gen*.pkl"))
    if not gen_files:
        return []


    def get_gen_num(filepath):
        match = re.search(r'-gen(\d+)_', os.path.basename(filepath))
        return int(match.group(1)) if match else -1

    gen_files.sort(key=get_gen_num)

    all_populations = []
    for f in gen_files:
        try:
            with open(f, "rb") as file:
                all_populations.append(pickle.load(file))
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Warning: Could not load {f}, it may be corrupted. Skipping. Error: {e}")

    print(f"Successfully loaded {len(all_populations)} individual generation files.")
    return all_populations


def flatten_generations_data(all_generations_list):
    """
    Takes the nested list output from `load_individual_generations` and flattens it
    into two NumPy arrays: one for all chromosomes and one for their fitness scores.

    This is the format required by the rule extraction script.
    """
    if not all_generations_list:
        return np.array([]), np.array([])

    population_list = []
    fitness_list = []

    for generation in all_generations_list:
        for individual in generation:

            chromosome_parts, fitness_score = individual

            flattened_chromosome = sum(chromosome_parts, [])
            population_list.append(flattened_chromosome)
            fitness_list.append(fitness_score)

    population_array = np.array(population_list, dtype=object)
    fitness_array = np.array(fitness_list)

    print(f"Successfully flattened {len(population_array)} individuals from {len(all_generations_list)} generations.")
    return population_array, fitness_array


def get_num_classes_from_generations(all_generations_list):
    """
    Robustly determines the number of classes by inspecting the structure
    of the first valid chromosome found in the population data.
    """
    if not all_generations_list or not all_generations_list[0]:
        print("Warning: Population data is empty, cannot determine number of classes.")
        return 0
    

    first_individual = all_generations_list[0][0]
    chromosome_parts = first_individual[0]
    data_part = chromosome_parts[0]
    


    num_classes = len(data_part) - 4
    
    print(f"Robustly detected num_classes = {num_classes} from chromosome structure.")
    return num_classes
