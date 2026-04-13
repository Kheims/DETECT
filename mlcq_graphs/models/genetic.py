"""Genetic algorithm search-based approach for code smell detection.

Evolves metric threshold rules using evolutionary optimization.
Each chromosome encodes threshold values for OO metrics.
Fitness is evaluated as multi-label F1 on the validation set.
"""

import random
import numpy as np
from sklearn.metrics import f1_score


class Individual:
    def __init__(self, num_metrics, num_labels):
        self.num_metrics = num_metrics
        self.num_labels = num_labels
        # Each label has its own set of thresholds + operator weights
        # Chromosome: for each label, threshold per metric + direction (above/below)
        self.thresholds = np.random.uniform(0, 1, (num_labels, num_metrics))
        self.directions = np.random.choice([1, -1], (num_labels, num_metrics))
        self.weights = np.random.uniform(0, 1, (num_labels, num_metrics))
        self.fitness = 0.0

    def predict(self, X):
        """Predict multi-label using threshold rules."""
        n_samples = X.shape[0]
        preds = np.zeros((n_samples, self.num_labels), dtype=int)
        for label_idx in range(self.num_labels):
            scores = np.zeros(n_samples)
            for m in range(self.num_metrics):
                if self.directions[label_idx, m] > 0:
                    match = (X[:, m] > self.thresholds[label_idx, m]).astype(float)
                else:
                    match = (X[:, m] < self.thresholds[label_idx, m]).astype(float)
                scores += self.weights[label_idx, m] * match
            threshold = np.sum(self.weights[label_idx]) * 0.5
            preds[:, label_idx] = (scores > threshold).astype(int)
        return preds

    def mutate(self, mutation_rate=0.1):
        for label_idx in range(self.num_labels):
            for m in range(self.num_metrics):
                if random.random() < mutation_rate:
                    self.thresholds[label_idx, m] += np.random.normal(0, 0.1)
                    self.thresholds[label_idx, m] = np.clip(self.thresholds[label_idx, m], 0, 1)
                if random.random() < mutation_rate:
                    self.directions[label_idx, m] *= -1
                if random.random() < mutation_rate:
                    self.weights[label_idx, m] += np.random.normal(0, 0.1)
                    self.weights[label_idx, m] = np.clip(self.weights[label_idx, m], 0, 1)


def crossover(parent1, parent2):
    child = Individual(parent1.num_metrics, parent1.num_labels)
    mask = np.random.random(parent1.thresholds.shape) > 0.5
    child.thresholds = np.where(mask, parent1.thresholds, parent2.thresholds)
    child.directions = np.where(mask, parent1.directions, parent2.directions)
    child.weights = np.where(mask, parent1.weights, parent2.weights)
    return child


def evaluate_fitness(individual, X, y):
    preds = individual.predict(X)
    individual.fitness = f1_score(y, preds, average="macro", zero_division=0)
    return individual.fitness


class GeneticSmellDetector:
    def __init__(self, num_metrics, num_labels=4, population_size=100,
                 generations=200, mutation_rate=0.1, crossover_rate=0.7,
                 tournament_size=5, elite_size=5, seed=42):
        self.num_metrics = num_metrics
        self.num_labels = num_labels
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.seed = seed
        self.best_individual = None

    def _tournament_select(self, population):
        candidates = random.sample(population, self.tournament_size)
        return max(candidates, key=lambda ind: ind.fitness)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Normalize features to [0, 1]
        self.min_vals = X_train.min(axis=0)
        self.max_vals = X_train.max(axis=0)
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1
        X_norm = (X_train - self.min_vals) / range_vals

        population = [Individual(self.num_metrics, self.num_labels)
                      for _ in range(self.population_size)]

        for gen in range(self.generations):
            for ind in population:
                evaluate_fitness(ind, X_norm, y_train)

            population.sort(key=lambda x: x.fitness, reverse=True)
            self.best_individual = population[0]

            if (gen + 1) % 20 == 0:
                print(f"  Generation {gen+1}/{self.generations} | Best F1-macro: {self.best_individual.fitness:.4f}")

            # Elitism
            new_pop = population[:self.elite_size]

            while len(new_pop) < self.population_size:
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)
                if random.random() < self.crossover_rate:
                    child = crossover(p1, p2)
                else:
                    child = Individual(self.num_metrics, self.num_labels)
                    child.thresholds = p1.thresholds.copy()
                    child.directions = p1.directions.copy()
                    child.weights = p1.weights.copy()
                child.mutate(self.mutation_rate)
                new_pop.append(child)

            population = new_pop

        return self

    def predict(self, X):
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1
        X_norm = (X - self.min_vals) / range_vals
        return self.best_individual.predict(X_norm)
