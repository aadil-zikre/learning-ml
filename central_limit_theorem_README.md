# Central Limit Theorem Demonstration

This Python code demonstrates the Central Limit Theorem using the NumPy, Seaborn, and Matplotlib libraries. The Central Limit Theorem states that the distribution of sample means approaches a normal distribution as the sample size increases, regardless of the shape of the original population distribution.

## Class: `CentralLimitTheorem`

The `CentralLimitTheorem` class encapsulates the functionality to generate samples, plot the distribution of sample means, and demonstrate the Central Limit Theorem.

### Initialization

The class constructor takes the following parameters:
- `population_mean` (default: 0): The mean of the population distribution.
- `population_std` (default: 1): The standard deviation of the population distribution.
- `sample_size` (default: 30): The size of each sample.
- `num_samples` (default: 1000): The number of samples to generate.

### Methods

#### `generate_samples(size, num_samples)`

This method generates `num_samples` samples, each of size `size`, from a normal distribution with the specified `population_mean` and `population_std`. It returns a list of sample means.

#### `plot_samples(samples, size, num_samples)`

This method plots a histogram of the sample means using Seaborn's `histplot` function. It sets the color of the KDE (Kernel Density Estimate) line to black. The plot includes a title, x-label, and y-label.

#### `demonstrate(vary_sample_size=False, vary_num_samples=False)`

This method demonstrates the Central Limit Theorem by varying either the sample size or the number of samples.

- If `vary_sample_size` is `True`, it varies the sample size from 10 to `sample_size` in increments of 10, while keeping the number of samples constant.
- If `vary_num_samples` is `True`, it varies the number of samples from 1000 to `num_samples` in increments of 1000, while keeping the sample size constant.
- If neither `vary_sample_size` nor `vary_num_samples` is `True`, it plots a single graph with the specified `sample_size` and `num_samples`.

The method uses interactive mode to update the plot dynamically. It clears the figure and plots the updated distribution of sample means for each iteration. There is a pause of 1 second after every 10 increments in sample size or 1000 increments in the number of samples.

## Usage

To use the code, create an instance of the `CentralLimitTheorem` class with the desired parameters:

```python
clt = CentralLimitTheorem(population_mean=10, population_std=2, sample_size=50, num_samples=50000)
```

Then, call the `demonstrate` method with the appropriate arguments to demonstrate the Central Limit Theorem:

```python
clt.demonstrate(vary_sample_size=True)  # Vary the sample size
clt.demonstrate(vary_num_samples=True)  # Vary the number of samples
clt.demonstrate()  # Plot a single graph with the specified sample size and number of samples
```

The code will display the plots showing the distribution of sample means for each demonstration.

## Requirements

- NumPy
- Seaborn
- Matplotlib

Make sure to have these libraries installed before running the code.

---

This code provides a visual demonstration of the Central Limit Theorem, allowing users to observe how the distribution of sample means approaches a normal distribution as the sample size or the number of samples increases.