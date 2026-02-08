import random
import numpy as np
import pickle
import os
import datetime
import json
from functools import lru_cache
from model_architectures import ModelBuilder
from trainer import ModelTrainer
import tensorflow as tf
import gc
import io
from contextlib import redirect_stdout
import traceback




def update_best_model_info(ga_instance, best_chromosome_parts, config, logger, run_dir):
    """
    Builds the current best model from a LIVE GA run, evaluates it,
    and updates the analysis context file. This is the correct implementation for live updates.
    """
    context_filepath = os.path.join(run_dir, "run_analysis_context.json")
    try:
        with open(context_filepath, 'r') as f:
            context = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Could not find analysis context file at {context_filepath}. Skipping update.")
        return

    logger.info("Updating analysis context with new best model information...")


    data_part = best_chromosome_parts[0]
    win_idx, over_idx = data_part[0], data_part[1]
    win_size_s = config.data_preprocessing.WINDOW_SIZE_SECONDS_OPTIONS[win_idx]
    overlap_r = config.data_preprocessing.OVERLAP_RATIO_OPTIONS[over_idx]
    data_filename = f"data_ws_{win_size_s}s_ol_{overlap_r}.npz"
    data_filepath = os.path.join(config.data_preprocessing.DATA_CACHE_DIR, data_filename)

    try:
        data_dict = np.load(data_filepath, allow_pickle=True)
    except FileNotFoundError:
        logger.error(f"Data file not found for best chromosome: {data_filename}. Cannot update context.")
        return


    hp1d, hp2d, hp_lstm, hp_mlp, loss, gamma, weights, batch = ga_instance._decode_chromosome(best_chromosome_parts)
    num_classes = int(data_dict["num_classes"])
    model_builder = ModelBuilder(
        input_shape_1d=data_dict["X_time_valid"].shape[1:],
        input_shape_2d=data_dict["X_freq_valid"].shape[1:],
        num_classes=num_classes, config=config, logger=logger
    )

    model = model_builder.build_model(hp1d, hp2d, hp_lstm, hp_mlp, loss, gamma, weights)


    summary_stream = io.StringIO()
    with redirect_stdout(summary_stream):
        model.summary(expand_nested=True)
    hyperparameters_str = pretty_print_chromosome(best_chromosome_parts, config, num_classes)

    trainer = ModelTrainer(config, logger)
    val_report, val_cm, test_report, test_cm = trainer.evaluate_on_validation_set(model, data_dict)


    context['BEST_MODEL_VALIDATION_REPORT'] = val_report
    context['BEST_MODEL_VALIDATION_CONFUSION'] = val_cm
    context['BEST_MODEL_TEST_REPORT'] = test_report
    context['BEST_MODEL_TEST_CONFUSION'] = test_cm
    context['BEST_MODEL_SUMMARY'] = summary_stream.getvalue()
    context['BEST_HYPERPARAMETERS'] = hyperparameters_str


    with open(context_filepath, 'w') as f:
        json.dump(context, f, indent=4)

    logger.info("Successfully updated analysis context file.")


    del model, model_builder, trainer
    tf.keras.backend.clear_session()
    gc.collect()







def pretty_print_chromosome(chromosome_parts, config, num_classes):
    """Converts a chromosome into a human-readable string representation including ALL hyperparameters."""
    output = []
    ns = config.network_structure
    active_streams = config.ga.ACTIVE_STREAMS




    data_part = chromosome_parts[0]
    win_idx, over_idx, loss_idx, gamma_idx = data_part[0:4]


    class_weight_genes = data_part[4:]
    class_weights_vals = [config.loss_function.CLASS_WEIGHT_MULTIPLIER_OPTIONS[g] for g in class_weight_genes]

    win_size = config.data_preprocessing.WINDOW_SIZE_SECONDS_OPTIONS[win_idx]
    overlap = config.data_preprocessing.OVERLAP_RATIO_OPTIONS[over_idx]
    loss_fn = "Categorical Crossentropy" if loss_idx == 0 else "Focal Loss"
    gamma = config.loss_function.GAMMA_OPTIONS[gamma_idx]


    batch_idx = chromosome_parts[-1][0]
    batch_size = config.ga.BATCH_SIZES_OPTIONS[batch_idx]

    output.append(f"--- 📊 Data & Training Configuration ---")
    output.append(f"  • Window Size:    {win_size}s")
    output.append(f"  • Overlap Ratio:  {int(overlap*100)}%")
    output.append(f"  • Batch Size:     {batch_size}")
    output.append(f"  • Loss Function:  {loss_fn}" + (f" (Gamma: {gamma})" if loss_idx == 1 else ""))


    cw_str = ", ".join([f"Cls{i}:{w}" for i, w in enumerate(class_weights_vals)])
    output.append(f"  • Class Weights:  [{cw_str}]")




    output.append("\n--- 🧠 Network Architecture ---")


    def print_conv_block(block_num, block_genes, prefix):
        if not block_genes[0]: return [f"  └─ {prefix} Block {block_num}: ❌ Inactive"]
        filters, kernel, act_idx = block_genes[1], block_genes[2], block_genes[3]
        activation = config.mappings.ACTIVATION_MAP[act_idx]
        strides, use_bn, use_pool = block_genes[5], bool(block_genes[6]), bool(block_genes[7])
        pool_type = "Max" if block_genes[8] == 0 else "Avg"

        details = f"Filters={filters}, Kernel={kernel}, Stride={strides}, Act={activation}"
        extras = []
        if use_bn: extras.append("BN")
        if use_pool: extras.append(f"{pool_type}Pool")

        return [f"  ├─ {prefix} Block {block_num}: ✅ {details} [{'|'.join(extras)}]"]

    def print_post_conv(genes):

        glob_pool = "Global Avg Pooling" if genes[0] else "Flatten"
        drop_str = f", Dropout={genes[2]:.2f}" if genes[1] else ""
        return [f"  └─ Output: {glob_pool}{drop_str}"]

    def print_lstm_block(block_num, block_genes):
        if not block_genes[0]: return [f"  └─ LSTM Block {block_num}: ❌ Inactive"]
        units, act_idx, rec_drop = block_genes[1], block_genes[2], block_genes[3]
        activation = config.mappings.ACTIVATION_MAP[act_idx]
        use_bn, use_drop, drop_rate = bool(block_genes[4]), bool(block_genes[5]), block_genes[6]

        details = f"Units={units}, Act={activation}, RecDrop={rec_drop:.2f}"
        extras = []
        if use_bn: extras.append("BN")
        if use_drop: extras.append(f"Drop={drop_rate:.2f}")

        return [f"  ├─ LSTM Block {block_num}: ✅ {details} [{'|'.join(extras)}]"]

    def print_mlp_block(block_num, block_genes):

        if not block_genes[0]: return [f"  └─ MLP Block {block_num}: ❌ Inactive"]
        units, act_idx, use_drop, drop_rate = block_genes[3], block_genes[4], bool(block_genes[5]), block_genes[6]
        activation = config.mappings.ACTIVATION_MAP[act_idx]

        details = f"Units={units}, Act={activation}"
        extras = []
        if bool(block_genes[1]): extras.append("BN")
        if use_drop: extras.append(f"Drop={drop_rate:.2f}")

        return [f"  ├─ MLP Block {block_num}: ✅ {details} [{'|'.join(extras)}]"]


    genes_per_conv = len(vars(ns.conv_block_bounds))
    genes_per_lstm = len(vars(ns.lstm_block_bounds))
    genes_per_mlp = len(vars(ns.mlp_block_bounds))


    if active_streams.cnn_1d:
        output.append("👉 Stream: 1D CNN (Time)")
        hp_part = chromosome_parts[1]
        for i in range(ns.N_CONV_BLOCKS_1D):
            start, end = i * genes_per_conv, (i + 1) * genes_per_conv
            output.extend(print_conv_block(i, hp_part[start:end], "Conv1D"))

        output.extend(print_post_conv(hp_part[ns.N_CONV_BLOCKS_1D * genes_per_conv:]))


    if active_streams.cnn_2d:
        output.append("👉 Stream: 2D CNN (Frequency)")
        hp_part = chromosome_parts[2]
        for i in range(ns.N_CONV_BLOCKS_2D):
            start, end = i * genes_per_conv, (i + 1) * genes_per_conv
            output.extend(print_conv_block(i, hp_part[start:end], "Conv2D"))
        output.extend(print_post_conv(hp_part[ns.N_CONV_BLOCKS_2D * genes_per_conv:]))


    if active_streams.lstm:
        output.append("👉 Stream: LSTM (Time)")
        hp_part = chromosome_parts[3]
        for i in range(ns.N_LSTM_BLOCKS):
            start, end = i * genes_per_lstm, (i + 1) * genes_per_lstm
            output.extend(print_lstm_block(i, hp_part[start:end]))


    output.append("👉 Shared MLP Head & Optimizer")
    hp_part = chromosome_parts[4]

    for i in range(ns.N_MLP_BLOCKS):
        start, end = i * genes_per_mlp, (i + 1) * genes_per_mlp
        output.extend(print_mlp_block(i, hp_part[start:end]))


    net_params_start = ns.N_MLP_BLOCKS * genes_per_mlp
    net_params = hp_part[net_params_start:]


    if len(net_params) >= 2:
        l1_reg = net_params[0]
        learning_rate = net_params[1]
        output.append(f"  🔹 L1 Regularization: {l1_reg:.6f}")
        output.append(f"  🔹 Learning Rate:     {learning_rate:.6f}")

    return "\n".join(output)


class ChromosomeHelper:
    def __init__(self, num_classes, config, logger):
        self.num_classes = num_classes
        self.config = config
        self.logger = logger


        self.varbound_conv1d = config.derived_ga_params.VARBOUND_CONV1D
        self.vartype_conv1d = config.derived_ga_params.VARTYPE_CONV1D
        self.varbound_conv2d = config.derived_ga_params.VARBOUND_CONV2D
        self.vartype_conv2d = config.derived_ga_params.VARTYPE_CONV2D
        self.varbound_hp_lstm = config.derived_ga_params.VARBOUND_HP_LSTM
        self.vartype_hp_lstm = config.derived_ga_params.VARTYPE_HP_LSTM
        self.varbound_mlp_shared = config.derived_ga_params.VARBOUND_MLP_SHARED
        self.vartype_mlp_shared = config.derived_ga_params.VARTYPE_MLP_SHARED



        self.len_data_fixed_part = 4
        self.len_data_chromosome = self.len_data_fixed_part + self.num_classes

    def get_class_weights_from_chromosome(self, data_part, cache_dir):
        class_weight_genes = data_part[self.len_data_fixed_part:]
        return np.array([self.config.loss_function.CLASS_WEIGHT_MULTIPLIER_OPTIONS[g] for g in class_weight_genes])

    def decode_hp1d_part(self, hp1d_part):
        return hp1d_part

    def decode_hp2d_part(self, hp2d_part):
        return hp2d_part


    def generate_conv1d_part(self):
        return [self._generate_gene(self.vartype_conv1d[i], self.varbound_conv1d[i]) for i in range(len(self.vartype_conv1d))]

    def generate_conv2d_part(self):
        return [self._generate_gene(self.vartype_conv2d[i], self.varbound_conv2d[i]) for i in range(len(self.vartype_conv2d))]

    def generate_hp_lstm_part(self):
            return [self._generate_gene(self.vartype_hp_lstm[i], self.varbound_hp_lstm[i]) for i in range(len(self.vartype_hp_lstm))]


    def generate_mlp_shared_part(self):
        return [self._generate_gene(self.vartype_mlp_shared[i], self.varbound_mlp_shared[i]) for i in range(len(self.vartype_mlp_shared))]


    def _generate_gene(self, vartype, varbound):
        if vartype == "int":
            return random.randint(varbound[0], varbound[1])
        elif vartype == "real":
            return random.uniform(varbound[0], varbound[1])
        return None

    def generate_data_chromosome_part(self):

        winsize_gene = random.randint(0, len(self.config.data_preprocessing.WINDOW_SIZE_SECONDS_OPTIONS) - 1)


        jmp_gene = random.randint(0, len(self.config.data_preprocessing.OVERLAP_RATIO_OPTIONS) - 1)


        loss_type_gene = random.randint(0, 1)


        gamma_gene = random.randint(0, len(self.config.loss_function.GAMMA_OPTIONS) - 1)

        data_genes = [winsize_gene, jmp_gene, loss_type_gene, gamma_gene]


        for _ in range(self.num_classes):
            data_genes.append(random.randint(0, len(self.config.loss_function.CLASS_WEIGHT_MULTIPLIER_OPTIONS) - 1))
        return data_genes


    def create_full_random_chromosome(self):
        """Creates a full, unified chromosome with all possible stream genes."""
        data_part = self.generate_data_chromosome_part()
        batch_idx_part = [random.randint(0, len(self.config.ga.BATCH_SIZES_OPTIONS) - 1)]

        active = self.config.ga.ACTIVE_STREAMS
        conv1d_part = self.generate_conv1d_part() if active.cnn_1d else []
        conv2d_part = self.generate_conv2d_part() if active.cnn_2d else []
        lstm_part = self.generate_hp_lstm_part() if active.lstm else []

        mlp_shared_part = self.generate_mlp_shared_part()

        return [data_part, conv1d_part, conv2d_part, lstm_part, mlp_shared_part, batch_idx_part]





    def _mutate_hp_part(self, hp_part, vartypes, varbounds):
        """Mutates a hyperparameter part on a per-gene basis."""
        mutated_part = hp_part[:]

        gene_mut_prob = 0.10

        for i in range(len(mutated_part)):
            if random.random() < gene_mut_prob:

                original_value = mutated_part[i]
                mutated_value = self._generate_gene(vartypes[i], varbounds[i])


                if vartypes[i] == "real" and random.random() < 0.8:
                    perturb_range = (varbounds[i][1] - varbounds[i][0]) * 0.1
                    perturbation = random.uniform(-perturb_range, perturb_range)
                    mutated_value = np.clip(original_value + perturbation, varbounds[i][0], varbounds[i][1])

                mutated_part[i] = mutated_value
        return mutated_part

    def mutate_data_part(self, data_part):
        """Mutates the data part on a per-gene basis."""
        mutated_part = data_part[:]
        gene_mut_prob = 0.10

        for i in range(len(mutated_part)):
            if random.random() < gene_mut_prob:
                if i == 0:
                    mutated_part[i] = random.randint(0, len(self.config.data_preprocessing.WINDOW_SIZE_SECONDS_OPTIONS) - 1)
                elif i == 1:
                    mutated_part[i] = random.randint(0, len(self.config.data_preprocessing.OVERLAP_RATIO_OPTIONS) - 1)
                elif i == 2:
                    mutated_part[i] = random.randint(0, 1)
                elif i == 3:
                    mutated_part[i] = random.randint(0, len(self.config.loss_function.GAMMA_OPTIONS) - 1)
                else:
                    mutated_part[i] = random.randint(0, len(self.config.loss_function.CLASS_WEIGHT_MULTIPLIER_OPTIONS) - 1)
        return mutated_part



    def _crossover_single_point(self, p1_part, p2_part):
        """The original single-point crossover."""
        if len(p1_part) <= 1: return p1_part[:], p2_part[:]
        point = random.randint(1, len(p1_part) - 1)
        c1_part = p1_part[:point] + p2_part[point:]
        c2_part = p2_part[:point] + p1_part[point:]
        return c1_part, c2_part

    def crossover_part(self, p1_part, p2_part, uniform_prob=0.9):
        """
        Performs crossover, randomly choosing between uniform and single-point
        to enhance diversity.
        """
        if len(p1_part) <= 1 or len(p1_part) != len(p2_part):
            return p1_part[:], p2_part[:]


        if random.random() < uniform_prob:
            c1_part, c2_part = [], []
            for i in range(len(p1_part)):
                if random.random() < 0.5:
                    c1_part.append(p1_part[i])
                    c2_part.append(p2_part[i])
                else:
                    c1_part.append(p2_part[i])
                    c2_part.append(p1_part[i])
            return c1_part, c2_part
        else:

            return self._crossover_single_point(p1_part, p2_part)


class GeneticAlgorithm:
    def __init__(self, chromosome_helper, config, logger, run_dir, add_generations=None):
        self.chromosome_helper = chromosome_helper
        self.config = config
        self.logger = logger
        self.run_dir = run_dir
        self.add_generations = add_generations

        self.population_size = self.config.ga.POPULATION_SIZE
        self.generations = self.config.ga.GENERATIONS

        self.model_trainer = ModelTrainer(self.config, self.logger)



        self.all_generations_population = []
        self.fitness_cache = {}



        self.max_children = int(self.population_size * self.config.ga.MAX_CHILDREN_PER_GENERATION_FACTOR)
        if self.max_children == 0 and self.population_size > 0 : self.max_children = 1
        self.crossover_prob = self.config.ga.CROSSOVER_PROBABILITY
        self.mutation_prob = self.config.ga.MUTATION_PROBABILITY
        self.n_trials = self.config.ga.N_TRIALS_PER_EVALUATION
        self.epochs = self.config.ga.EPOCHS_PER_TRIAL
        self.fit_verbose = self.config.ga.FITNESS_VERBOSE_LEVEL
        self.patience = self.config.ga.PATIENCE_EARLY_STOPPING
        self.batch_size_options = self.config.ga.BATCH_SIZES_OPTIONS

        self.population = []
        self.best_fitness_history = []

    def _save_config(self):
        """Saves the configuration to a JSON file in the run directory."""
        config_path = os.path.join(self.run_dir, 'config.json')

        def sns_to_dict(sns):
            if isinstance(sns, list):
                return [sns_to_dict(item) for item in sns]
            if not hasattr(sns, '__dict__'):
                return sns
            return {key: sns_to_dict(value) for key, value in sns.__dict__.items()}

        with open(config_path, 'w') as f:
            json.dump(sns_to_dict(self.config), f, indent=4)
        self.logger.info(f"Configuration saved to {config_path}")




    def apply_mutation(self, chromosome_parts):
        """Applies mutation to all parts of a chromosome."""
        mutated_parts = [part[:] for part in chromosome_parts]


        if random.random() < self.mutation_prob:
            mutated_parts[0] = self.chromosome_helper.mutate_data_part(mutated_parts[0])


        if random.random() < self.mutation_prob:
            mutated_parts[1] = self.chromosome_helper._mutate_hp_part(mutated_parts[1], self.chromosome_helper.vartype_conv1d, self.chromosome_helper.varbound_conv1d)
        if random.random() < self.mutation_prob:
            mutated_parts[2] = self.chromosome_helper._mutate_hp_part(mutated_parts[2], self.chromosome_helper.vartype_conv2d, self.chromosome_helper.varbound_conv2d)
        if random.random() < self.mutation_prob:
             mutated_parts[3] = self.chromosome_helper._mutate_hp_part(mutated_parts[3], self.chromosome_helper.vartype_hp_lstm, self.chromosome_helper.varbound_hp_lstm)
        if random.random() < self.mutation_prob:
            mutated_parts[4] = self.chromosome_helper._mutate_hp_part(mutated_parts[4], self.chromosome_helper.vartype_mlp_shared, self.chromosome_helper.varbound_mlp_shared)
        if random.random() < self.mutation_prob:
            mutated_parts[5] = [random.randint(0, len(self.batch_size_options) - 1)]

        return mutated_parts


    def _decode_chromosome(self, chromosome_parts):
        """Decodes the unified chromosome structure."""
        data_part, hp1d, hp2d, hp_lstm, hp_mlp, batch_idx_part = chromosome_parts

        batch_size = self.batch_size_options[batch_idx_part[0]]
        loss_gene = data_part[2]
        gamma_gene = data_part[3]
        gamma = self.config.loss_function.GAMMA_OPTIONS[gamma_gene]
        class_weights = self.chromosome_helper.get_class_weights_from_chromosome(data_part, self.config.data_preprocessing.DATA_CACHE_DIR)

        return hp1d, hp2d, hp_lstm, hp_mlp, loss_gene, gamma, class_weights, batch_size


    def _calculate_fitness(self, chromosome_tuple):
        chromosome_parts = [list(part) for part in chromosome_tuple]
        self.logger.debug(f"Evaluating fitness for chromosome (hash: {hash(chromosome_tuple)})")

        data_part = chromosome_parts[0]
        win_size_idx, overlap_idx = data_part[0], data_part[1]
        win_size_s = self.config.data_preprocessing.WINDOW_SIZE_SECONDS_OPTIONS[win_size_idx]
        overlap_r = self.config.data_preprocessing.OVERLAP_RATIO_OPTIONS[overlap_idx]
        data_filename = f"data_ws_{win_size_s}s_ol_{overlap_r}.npz"
        data_filepath = os.path.join(self.config.data_preprocessing.DATA_CACHE_DIR, data_filename)

        try:
            loaded_data = np.load(data_filepath, allow_pickle=True)
            self.logger.info(f"Successfully loaded data from {data_filepath}")
        except FileNotFoundError:
            self.logger.error(f"Data file not found: {data_filepath}. Returning 0 fitness.")
            return 0.0

        temp_data_dict = {key: loaded_data[key] for key in loaded_data}


        hp1d, hp2d, hp_lstm, hp_mlp, loss_gene, gamma, class_weights, batch_size = self._decode_chromosome(chromosome_parts)

        total_f1_score = 0
        if temp_data_dict["y_dev"].shape[0] == 0 or temp_data_dict["y_valid"].shape[0] == 0:
            self.logger.warning("Training or validation data is empty. Returning 0 fitness.")
            return 0.0

        for trial in range(self.n_trials):
            self.logger.info(f"Fitness trial {trial + 1}/{self.n_trials}")
            model_builder = ModelBuilder(
                    input_shape_1d=temp_data_dict["X_time_dev"].shape[1:],
                    input_shape_2d=temp_data_dict["X_freq_dev"].shape[1:],
                    num_classes=int(temp_data_dict["num_classes"]),
                    config=self.config,
                    logger=self.logger
            )
            try:
                model = model_builder.build_model(
                    hp1d=hp1d, hp2d=hp2d, hp_lstm=hp_lstm, hp_mlp=hp_mlp, loss_gene=loss_gene,
                    gamma=gamma, class_weights=class_weights
                )
            except Exception as e:
                self.logger.debug(f"Error building model: {e}. Returning 0 fitness.")
                return 0.0

            try:
                f1 = self.model_trainer.train_and_evaluate(
                    model, temp_data_dict, batch_size, self.epochs,
                    loss_gene, class_weights, self.fit_verbose, self.run_dir
                )
            except Exception as e:
                self.logger.error(f"Error training the trainer: {e}. Setting F1 to zero.")
                f1 = 0.0

            total_f1_score += f1
            del model, model_builder
            tf.keras.backend.clear_session()
            gc.collect()

        avg_f1_score = total_f1_score / self.n_trials if self.n_trials > 0 else 0
        self.logger.info(f"==> Evaluated Fitness: {avg_f1_score:.4f}")
        return avg_f1_score

    def _get_fitness(self, chromosome_parts):
        """Wrapper to handle caching for the fitness function."""
        chromosome_tuple = tuple(map(tuple, chromosome_parts))
        if chromosome_tuple in self.fitness_cache:
            self.logger.debug("Fitness cache HIT.")
            return self.fitness_cache[chromosome_tuple]

        self.logger.debug("Fitness cache MISS.")
        fitness = self._calculate_fitness(chromosome_tuple)
        self.fitness_cache[chromosome_tuple] = fitness
        return fitness

    def initialize_population(self):
        self.logger.info(f"Initializing population of size {self.population_size}")

        try:
            sample_win_size = self.config.data_preprocessing.WINDOW_SIZE_SECONDS_OPTIONS[0]
            sample_overlap = self.config.data_preprocessing.OVERLAP_RATIO_OPTIONS[0]
            sample_data_filename = f"data_ws_{sample_win_size}s_ol_{sample_overlap}.npz"
            sample_data_filepath = os.path.join(self.config.data_preprocessing.DATA_CACHE_DIR, sample_data_filename)
            sample_data = np.load(sample_data_filepath, allow_pickle=True)
            self.logger.info(f"Loaded sample data for initialization from {sample_data_filepath}")

            n_train_sample = len(sample_data["X_time_train"])
            n_dev_sample = int(n_train_sample * self.config.ga.DEV_RATIO)
            if n_dev_sample == 0 and n_train_sample > 0: n_dev_sample = 1

            if n_train_sample > 0:
                dev_indices_sample = np.random.choice(n_train_sample, size=n_dev_sample, replace=False)
                sample_x_time_dev_shape = sample_data["X_time_train"][dev_indices_sample].shape[1:]
                sample_x_freq_dev_shape = sample_data["X_freq_train"][dev_indices_sample].shape[1:]
            else:
                self.logger.warning("Sample training data is empty. Using validation data shapes.")
                sample_x_time_dev_shape = sample_data["X_time_valid"].shape[1:]
                sample_x_freq_dev_shape = sample_data["X_freq_valid"].shape[1:]

            sample_num_classes = int(sample_data["num_classes"])

        except Exception as e:
            self.logger.error(f"Could not load sample data for initialization. Error: {e}. Population initialization will fail.")
            return

        self.population = []
        while len(self.population) < self.population_size:
            chromosome_parts = self.chromosome_helper.create_full_random_chromosome()

            try:
                _hp1d, _hp2d, _hp_lstm, _hp_mlp, _, _, _, _ = self._decode_chromosome(chromosome_parts)
                temp_model_builder = ModelBuilder(
                    input_shape_1d=sample_x_time_dev_shape,
                    input_shape_2d=sample_x_freq_dev_shape,
                    num_classes=sample_num_classes,
                    config=self.config,
                    logger=self.logger
                )

                temp_model_builder.build_model(_hp1d, _hp2d, _hp_lstm, _hp_mlp, 0, 2.0, np.ones(sample_num_classes))
                tf.keras.backend.clear_session()
            except Exception as e:
                self.logger.warning(f"Initial chromosome failed build check. Retrying generation.")







                continue

            fitness = self._get_fitness(chromosome_parts)
            self.population.append([chromosome_parts, fitness])
            self.logger.info(f"Initialized individual {len(self.population)}/{self.population_size} with fitness {fitness:.4f}")

        self.population.sort(key=lambda x: x[1], reverse=True)

    def select_parents(self):
        tournament_size = max(2, self.population_size // 5)
        if not self.population: return None, None

        best_p1, best_p2 = None, None

        tournament_p1 = random.sample(self.population, tournament_size)
        tournament_p1.sort(key=lambda x: x[1], reverse=True)
        best_p1 = tournament_p1[0][0]

        while True:
            tournament_p2 = random.sample(self.population, tournament_size)
            tournament_p2.sort(key=lambda x: x[1], reverse=True)
            current_best_p2_parts = tournament_p2[0][0]
            if current_best_p2_parts != best_p1 or self.population_size <=1 :
                best_p2 = current_best_p2_parts
                break
            if tournament_size == self.population_size and self.population_size > 1:
                self.logger.warning("Could not find a different parent 2, re-using parent 1. Population might be converging.")
                best_p2 = best_p1
                break

        return best_p1, best_p2





    def _save_checkpoint(self, generation, patience_counter):
        """Saves the current state of the GA to a checkpoint file."""
        checkpoint_path = os.path.join(self.run_dir, "ga_checkpoint.pkl")
        state = {
            'generation': generation,
            'population': self.population,
            'best_fitness_history': self.best_fitness_history,
            'patience_counter': patience_counter,
            'cache_dump': self.fitness_cache,
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
        }
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(state, f)
            self.logger.info(f"Successfully saved GA checkpoint for generation {generation} to {checkpoint_path}")



            if self.population:
                best_individual_path = os.path.join(self.run_dir, "best_individual.pkl")
                with open(best_individual_path, "wb") as f:

                    pickle.dump(self.population[0], f)
                self.logger.info(f"Updated best_individual.pkl with fitness {self.population[0][1]:.4f}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self):
        """Loads the GA state from a checkpoint file if it exists."""
        checkpoint_path = os.path.join(self.run_dir, "ga_checkpoint.pkl")
        if not os.path.exists(checkpoint_path):
            self.logger.info("No checkpoint file found. Starting a new GA run.")
            return 0, 0

        try:
            with open(checkpoint_path, "rb") as f:
                state = pickle.load(f)

            start_gen = state['generation'] + 1
            self.population = state['population']
            self.best_fitness_history = state['best_fitness_history']
            patience_counter = state['patience_counter']
            random.setstate(state['random_state'])
            np.random.set_state(state['numpy_random_state'])


            self.fitness_cache = state.get('cache_dump', {})


            self.logger.info(f"Successfully loaded GA state from checkpoint. Resuming from generation {start_gen}.")
            self.logger.info(f"Restored population size: {len(self.population)}. Best fitness so far: {self.best_fitness_history[-1]:.4f}")
            self.logger.info(f"Restored fitness cache with {len(self.fitness_cache)} entries.")
            return start_gen, patience_counter

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}. Starting a new run.")


            return 0, 0

    def crossover_and_mutate(self, p1_parts, p2_parts):
        """Performs crossover on all parts, then applies mutation to children."""
        c1_parts, c2_parts = [], []


        for i in range(len(p1_parts)):
            c1_part, c2_part = self.chromosome_helper.crossover_part(p1_parts[i], p2_parts[i])
            c1_parts.append(c1_part)
            c2_parts.append(c2_part)


        c1_parts = self.apply_mutation(c1_parts)
        c2_parts = self.apply_mutation(c2_parts)

        c1_fitness = self._get_fitness(c1_parts)
        c2_fitness = self._get_fitness(c2_parts)

        return [[c1_parts, c1_fitness], [c2_parts, c2_fitness]]

    def run(self):
        self.logger.info("Starting Genetic Algorithm...")
        self._save_config()


        archive_checkpoints_dir = os.path.join(self.run_dir, "archive_checkpoints")
        os.makedirs(archive_checkpoints_dir, exist_ok=True)

        start_gen, patience_counter = self._load_checkpoint()

        if self.add_generations is not None and start_gen > 0:


            new_target_generation = start_gen + self.add_generations
            self.generations = new_target_generation
            self.logger.info(f"EXTENDING RUN by {self.add_generations} generations. New target generation count: {self.generations}")

        if start_gen == 0:

            self.initialize_population()
            if not self.population:
                self.logger.error("Population initialization failed. Exiting.")
                return None, False
            self.best_fitness_history.append(self.population[0][1])
            self.logger.info(f"Initial best fitness: {self.best_fitness_history[0]:.4f}")
            self._save_checkpoint(generation=0, patience_counter=patience_counter)
            update_best_model_info(
                ga_instance=self,
                best_chromosome_parts=self.population[0][0],
                config=self.config,
                logger=self.logger,
                run_dir=self.run_dir
            )

        for gen in range(start_gen, self.generations):
            self.logger.info(f"--- Generation {gen + 1}/{self.generations} ---")


            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_filename = os.path.join(archive_checkpoints_dir,
                                          f"{self.config.general.DATANAME_PREFIX}-gen{gen}_{timestamp}.pkl")
            with open(archive_filename, "wb") as f:
                pickle.dump(self.population, f)
            self.logger.info(f"Archived population for start of generation {gen + 1} to {archive_filename}")

            existing_chromosomes = {tuple(map(tuple, ind[0])) for ind in self.population}

            new_children = []
            attempts = 0
            max_attempts = self.max_children * 5


            while len(new_children) < self.max_children and attempts < max_attempts:
                p1_parts, p2_parts = self.select_parents()
                if p1_parts is None or p2_parts is None:
                    attempts += 1
                    continue


                if random.random() < self.crossover_prob:
                    children = self.crossover_and_mutate(p1_parts, p2_parts)
                else:

                    child1_mutated = self.apply_mutation(p1_parts)
                    child2_mutated = self.apply_mutation(p2_parts)
                    children = [
                        [child1_mutated, self._get_fitness(child1_mutated)],
                        [child2_mutated, self._get_fitness(child2_mutated)]
                    ]



                for child_parts, fitness in children:
                    child_tuple = tuple(map(tuple, child_parts))
                    if child_tuple not in existing_chromosomes:
                        new_children.append([child_parts, fitness])
                        existing_chromosomes.add(child_tuple)
                        self.logger.info(f"Generated a unique child with fitness {fitness:.4f}")

                        if len(new_children) >= self.max_children:
                            break
                attempts += 1

            if attempts >= max_attempts:
                self.logger.warning(f"Could not generate enough unique children after {max_attempts} attempts. Population may be stagnating.")


            self.population.extend(new_children)
            self.population.sort(key=lambda x: x[1], reverse=True)
            self.population = self.population[:self.population_size]
            self.all_generations_population.append(list(self.population))

            current_best_fitness = self.population[0][1]


            #



            #


            new_best_found = len(self.best_fitness_history) == 0 or current_best_fitness > self.best_fitness_history[-1]

            if new_best_found:
                 previous_best = self.best_fitness_history[-1] if self.best_fitness_history else 0.0
                 self.logger.info(f"✨ New Best Fitness Found: {current_best_fitness:.4f} (previously {previous_best:.4f}) ✨")
                 self.best_fitness_history.append(current_best_fitness)
                 update_best_model_info(self, self.population[0][0], self.config, self.logger, self.run_dir)




            self.logger.info(f"Generation {gen + 1} best fitness: {current_best_fitness:.4f}")
            self.logger.info(f"Top 3 fitnesses: {[round(ind[1], 4) for ind in self.population[:3]]}")

            best_chromosome = self.population[0][0]
            pretty_string = pretty_print_chromosome(best_chromosome, self.config, self.chromosome_helper.num_classes)
            self.logger.info(f"--- Best Architecture in Gen {gen + 1} ---\n{pretty_string}\n---------------------------------")


            if len(self.best_fitness_history) > 1 and current_best_fitness > self.best_fitness_history[-2]:
                patience_counter = 0
            else:
                patience_counter += 1


            self._save_checkpoint(generation=gen, patience_counter=patience_counter)

            if patience_counter >= self.patience:
                self.logger.info(f"Early stopping at generation {gen + 1} due to no improvement for {self.patience} generations.")
                break


        self.logger.info("Genetic Algorithm search phase finished.")
        self.logger.info(f"Final Best Fitness Achieved: {self.population[0][1]:.4f}")
        self.logger.info("--- Final Best Architecture Found by GA ---")
        best_chromosome_final = self.population[0][0]
        final_pretty_string = pretty_print_chromosome(best_chromosome_final, self.config, self.chromosome_helper.num_classes)
        self.logger.info(f"\n{final_pretty_string}\n" + "-"*45)



        final_pop_filename = os.path.join(self.run_dir, f"{self.config.general.DATANAME_PREFIX}-final_population.pkl")
        with open(final_pop_filename, "wb") as f:
            pickle.dump(self.population, f)
        self.logger.info(f"Saved final population to {final_pop_filename}")

        all_pops_filename = os.path.join(self.run_dir,
                                 f"{self.config.general.DATANAME_PREFIX}-all_generations_population.pkl")
        with open(all_pops_filename, "wb") as f:
            pickle.dump(self.all_generations_population, f)
        self.logger.info(f"Saved all generations population data to {all_pops_filename}")


        return self.population[0], True
