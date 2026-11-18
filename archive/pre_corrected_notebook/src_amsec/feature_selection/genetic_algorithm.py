"""Genetic feature selector transcribed from notebook cells 2/4."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class GeneticFeatureSelector:
    """Notebook hyperparameters: pop=40, generations=30, mutation=0.12, crossover=0.9."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        population_size: int = 40,
        generations: int = 30,
        mutation_rate: float = 0.12,
        crossover_rate: float = 0.9,
        n_estimators: int = 100,
        random_state: int = 42,
        progress_callback=None,
    ):
        self.X = np.asarray(x)
        self.y = np.asarray(y)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_features = self.X.shape[1]
        self.population: list[np.ndarray] = []
        self.fitness_history: list[float] = []
        self.best_individual: np.ndarray | None = None
        self.best_fitness = -np.inf
        self.progress_callback = progress_callback
        self._rng = np.random.RandomState(random_state)

    def initialize_population(self) -> None:
        self.population = []
        floor = max(3, int(self.n_features * 0.25))
        for _ in range(self.population_size):
            individual = self._rng.choice([0, 1], size=self.n_features, p=[0.25, 0.75])
            if int(individual.sum()) < floor:
                picks = self._rng.choice(self.n_features, floor, replace=False)
                individual[picks] = 1
            self.population.append(individual.astype(np.int8))

    def fitness_function(self, individual: np.ndarray) -> float:
        selected = np.flatnonzero(individual)
        if selected.size < 3:
            return 0.0
        x_sel = self.X[:, selected]
        x_train, x_val, y_train, y_val = train_test_split(
            x_sel, self.y, test_size=0.25, random_state=self.random_state, stratify=self.y
        )
        forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=14,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        forest.fit(x_train, y_train)
        predicted = forest.predict(x_val)
        acc = accuracy_score(y_val, predicted)
        f1 = f1_score(y_val, predicted, average="weighted", zero_division=0)
        penalty = 1.0 - (selected.size / self.n_features) * 0.06
        return float((0.55 * acc + 0.45 * f1) * penalty)

    def selection(self, fitness_scores: np.ndarray) -> list[np.ndarray]:
        selected = []
        for _ in range(self.population_size):
            tournament_idx = self._rng.choice(self.population_size, 4, replace=False)
            winner_idx = tournament_idx[int(np.argmax(fitness_scores[tournament_idx]))]
            selected.append(self.population[winner_idx].copy())
        return selected

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._rng.random() < self.crossover_rate:
            point = self._rng.randint(1, self.n_features)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        for i in range(self.n_features):
            if self._rng.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        if individual.sum() < 3:
            picks = self._rng.choice(self.n_features, 3, replace=False)
            individual[picks] = 1
        return individual

    def run(self) -> tuple[np.ndarray, list[float]]:
        self.initialize_population()
        for generation in range(self.generations):
            scores = np.array([self.fitness_function(ind) for ind in self.population])
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > self.best_fitness:
                self.best_fitness = float(scores[best_idx])
                self.best_individual = self.population[best_idx].copy()
            self.fitness_history.append(self.best_fitness)
            if self.progress_callback:
                self.progress_callback({"phase": "ga", "generation": generation + 1, "best_fitness": self.best_fitness})
            parents = self.selection(scores)
            next_pop: list[np.ndarray] = []
            for i in range(0, self.population_size, 2):
                p1 = parents[i]
                p2 = parents[i + 1 if i + 1 < self.population_size else 0]
                c1, c2 = self.crossover(p1, p2)
                next_pop.append(self.mutate(c1))
                if len(next_pop) < self.population_size:
                    next_pop.append(self.mutate(c2))
            # elitism
            next_pop[0] = self.best_individual.copy()
            self.population = next_pop
        selected = np.flatnonzero(self.best_individual)
        return selected, self.fitness_history
