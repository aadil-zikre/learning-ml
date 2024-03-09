import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class CentralLimitTheorem:
    def __init__(self, population_mean=0, population_std=1, sample_size=30, num_samples=1000):
        self.population_mean = population_mean
        self.population_std = population_std
        self.sample_size = sample_size
        self.num_samples = num_samples

    def generate_samples(self, size, num_samples):
        samples = []
        for _ in range(num_samples):
            sample = np.random.normal(self.population_mean, self.population_std, size)
            sample_mean = np.mean(sample)
            samples.append(sample_mean)
        return samples

    def plot_samples(self, samples, size, num_samples):
        ax = sns.histplot(samples, kde=True)
        ax.lines[-1].set_color('black')
        plt.title(f"Distribution of Sample Means (Sample Size: {size}, Number of Samples: {num_samples})")
        plt.xlabel("Sample Mean")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def demonstrate(self, vary_sample_size=False, vary_num_samples=False):
        if vary_sample_size:
            sample_sizes = range(10, self.sample_size + 1, 10)
            num_samples = self.num_samples
        elif vary_num_samples:
            sample_sizes = [self.sample_size]
            num_samples_range = range(1000, self.num_samples + 1, 1000)
        else:
            sample_sizes = [self.sample_size]
            num_samples = self.num_samples

        plt.ion()  # Enable interactive mode
        plt.figure(figsize=(8, 6))

        if vary_sample_size:
            for size in sample_sizes:
                samples = self.generate_samples(size, num_samples)
                plt.clf()  # Clear the current figure
                self.plot_samples(samples, size, num_samples)

                if size % 10 == 0:
                    plt.pause(1)  
        elif vary_num_samples:
            for num_samples in num_samples_range:
                samples = self.generate_samples(self.sample_size, num_samples)
                # plt.clf()  # Clear the current figure
                self.plot_samples(samples, self.sample_size, num_samples)

                if num_samples % 1000 == 0:
                    plt.pause(1)  
        else:
            samples = self.generate_samples(self.sample_size, self.num_samples)
            self.plot_samples(samples, self.sample_size, self.num_samples)

        plt.ioff()  # Disable interactive mode
        plt.show()  # Keep the final plot displayed

# Usage
if __name__ == "__main__":

    clt = CentralLimitTheorem(population_mean=10, population_std=2, sample_size=50, num_samples=50000)


    clt.demonstrate(vary_sample_size=True)


    clt.demonstrate(vary_num_samples=True)


    clt.demonstrate()