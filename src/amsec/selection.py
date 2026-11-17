"""Mutual-information prefilter and genetic feature search.

The cell-4 notebook never printed the five selected column names. A local
search therefore cannot be certified as the Kaggle subset. The search below
uses the same fitness definition on whatever matrix it is given.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class GeneticFeatureSelector:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        population_size: int = 20,
        generations: int = 8,
        mutation_rate: float = 0.12,
        crossover_rate: float = 0.9,
        n_estimators: int = 40,
        random_state: int = 42,
    ):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_features = self.x.shape[1]
        self.population: list[np.ndarray] = []
        self.fitness_history: list[float] = []
        self.best_individual: np.ndarray | None = None
        self.best_fitness = -np.inf
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

    def fitness(self, individual: np.ndarray) -> float:
        selected = np.flatnonzero(individual)
        if selected.size < 3:
            return 0.0
        x_sel = self.x[:, selected]
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
        accuracy = accuracy_score(y_val, predicted)
        f1 = f1_score(y_val, predicted, average="weighted", zero_division=0)
        penalty = 1.0 - (selected.size / self.n_features) * 0.06
        return float((0.55 * accuracy + 0.45 * f1) * penalty)

    def run(self) -> tuple[np.ndarray, list[float]]:
        self.initialize_population()
        for generation in range(self.generations):
            scores = np.array([self.fitness(ind) for ind in self.population])
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > self.best_fitness:
                self.best_fitness = float(scores[best_idx])
                self.best_individual = self.population[best_idx].copy()
            self.fitness_history.append(float(scores[best_idx]))
            print(
                f"  GA gen {generation + 1}/{self.generations}: "
                f"best_this_gen={scores[best_idx]:.4f} "
                f"kept={int(self.best_individual.sum())}",
                flush=True,
            )
            selected_pop = []
            for _ in range(self.population_size):
                tournament = self._rng.choice(self.population_size, 4, replace=False)
                winner = tournament[int(np.argmax(scores[tournament]))]
                selected_pop.append(self.population[winner].copy())
            children: list[np.ndarray] = []
            for i in range(0, self.population_size, 2):
                parent_a = selected_pop[i]
                parent_b = selected_pop[min(i + 1, self.population_size - 1)]
                if self._rng.rand() < self.crossover_rate and self.n_features > 1:
                    point = int(self._rng.randint(1, self.n_features))
                    child_a = np.concatenate([parent_a[:point], parent_b[point:]])
                    child_b = np.concatenate([parent_b[:point], parent_a[point:]])
                else:
                    child_a, child_b = parent_a.copy(), parent_b.copy()
                children.extend([self._mutate(child_a), self._mutate(child_b)])
            children = children[: self.population_size]
            children[0] = self.best_individual.copy()
            self.population = children
        assert self.best_individual is not None
        return np.flatnonzero(self.best_individual), self.fitness_history

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        for i in range(self.n_features):
            if self._rng.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        if int(individual.sum()) == 0:
            individual[int(self._rng.randint(0, self.n_features))] = 1
        return individual
