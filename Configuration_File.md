# NEAT Configuration File Explanation

This file describes the key elements configured in the `neat_config.txt` used for neuroevolution in this project. It includes common value ranges, default behaviors, and the effects of increasing or decreasing each parameter.


## [NEAT]

### `fitness_criterion`
Defines the metric used to compare genomes. Common values include `max`, `min`, or `mean`.
- **Typical:** `max` for maximizing fitness.
- **Effect:** Using `max` promotes selection of the best-performing individual in each generation.

### `fitness_threshold`
Specifies the fitness value at which evolution will stop.
- **Typical range:** 0.90 to 1.00
- **Lower value:** Stops evolution earlier, possibly before optimal solutions.
- **Higher value:** Requires better solutions, increasing runtime.

### `pop_size`
Number of individuals in the population.
- **Typical range:** 50 to 500
- **Lower values:** Faster computation but less genetic diversity.
- **Higher values:** Slower training but potentially better exploration and solutions.

### `reset_on_extinction`
Resets population if all species go extinct.
- **Recommended:** `True` for safety in unstable environments.
- **False:** May lead to complete stagnation if extinction occurs.


## [DefaultGenome]

### `feed_forward`
Determines if the network is feedforward (`true`) or recurrent (`false`).
- **Feedforward:** Simpler and faster to train.
- **Recurrent:** Can model temporal dependencies but are harder to optimize.

### `num_inputs`, `num_outputs`, `num_hidden`
Specify the number of input, output, and initial hidden nodes.
- **Typical:** Inputs match feature size; outputs depend on task (1 for binary).
- **Hidden = 0:** NEAT will evolve hidden structure from scratch.

### `initial_connection`
Defines how connections are initialized.
- **Typical values:** `full`, `partial`, or `sparse`
- **Full:** Every input is connected to every output at the start.


## Activation & Aggregation Functions

### `activation_default`, `activation_options`
Specifies the default and allowed activation functions.
- **Common functions:** `sigmoid`, `tanh`, `relu`
- **Effect:** Determines neuron output behavior. Sigmoid/tanh are smooth and bounded; relu allows sparse activations.

### `activation_mutate_rate`
Chance of mutating a neuron's activation function.
- **Typical range:** 0.0 to 0.3
- **Higher values:** More diversity but less stability.

### `aggregation_default`, `aggregation_options`
Defines how multiple inputs to a neuron are aggregated.
- **Options:** `sum`, `mean`, `max`, `min`


## Bias, Weights, and Response Parameters

### `bias_*`, `weight_*`, `response_*`
Control initialization, mutation, and range of biases, connection weights, and response parameters.
- **`*_mutate_rate`**: Probability of mutation.
- **`*_mutate_power`**: Standard deviation of the change applied.
- **`*_replace_rate`**: Chance to replace value instead of mutating it.
- **Typical ranges:**
  - Mutation rate: 0.5 to 0.9
  - Replace rate: 0.0 to 0.2
  - Value range: -5.0 to +5.0 or larger for expressive models


## Gene Enabling

### `enabled_default`
Whether new connections are enabled by default.
- **True:** New connections are active.
- **False:** Allows more controlled activation via mutation.

### `enabled_mutate_rate`
Probability of toggling the enabled/disabled status of a connection.
- **Typical:** 0.01 to 0.1
- **Higher values:** More frequent reactivation of dormant genes.



## Structural Mutation Rates

### `node_add_prob`, `node_delete_prob`
Control the probability of adding or removing nodes.
- **Typical range:** 0.1 to 0.5
- **Higher values:** Faster structural evolution, may reduce stability.

### `conn_add_prob`, `conn_delete_prob`
Probability of adding/removing connections between nodes.
- **Higher `conn_add_prob`:** Promotes faster network connectivity growth.
- **Higher `conn_delete_prob`:** Encourages simplification.


## [DefaultSpeciesSet]

### `compatibility_threshold`
Controls how genomes are grouped into species.
- **Typical range:** 2.0 to 5.0
- **Higher values:** More species (less sharing).
- **Lower values:** Fewer species (more sharing).


## [DefaultStagnation]

### `species_fitness_func`
Function used to track species performance (e.g., `max`, `mean`).
- **Max:** Focuses on the best genome within a species.

### `max_stagnation`
Maximum generations a species can go without improvement.
- **Typical:** 5 to 20
- **Lower values:** Prunes non-improving species quickly.

### `species_elitism`
Number of top-performing species preserved without modification.
- **Typical:** 1 to 5


## [DefaultReproduction]

### `elitism`
Number of top genomes preserved each generation.
- **Typical range:** 1 to 10
- **Higher values:** Preserves strong performers but reduces exploration.

### `survival_threshold`
Fraction of individuals within a species allowed to reproduce.
- **Typical:** 0.2 to 0.5
- **Lower values:** Strong selection pressure.
- **Higher values:** Promotes diversity but may slow convergence.
