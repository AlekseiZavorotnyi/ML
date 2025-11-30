import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    sorted_values = np.unique(feature_vector)
    if sorted_values.size < 2:
        return np.array([]), np.array([]), None, None

    candidate_thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2
    n_objects = len(target_vector)

    comparison_matrix = feature_vector.reshape(-1, 1) < candidate_thresholds.reshape(1, -1)

    left_group_sizes = np.sum(comparison_matrix, axis=0)
    right_group_sizes = n_objects - left_group_sizes

    valid_partitions = (left_group_sizes > 0) & (right_group_sizes > 0)

    if not np.any(valid_partitions):
        return candidate_thresholds, np.full(candidate_thresholds.shape, -np.inf), None, None

    positive_left = np.sum(target_vector.reshape(-1, 1) * comparison_matrix, axis=0)
    positive_right = np.sum(target_vector) - positive_left

    frac_pos_left = np.divide(positive_left, left_group_sizes,
                            where=left_group_sizes > 0,
                            out=np.zeros_like(positive_left, dtype=float))
    frac_pos_right = np.divide(positive_right, right_group_sizes,
                             where=right_group_sizes > 0,
                             out=np.zeros_like(positive_right, dtype=float))

    impurity_left = 1 - frac_pos_left**2 - (1 - frac_pos_left)**2
    impurity_right = 1 - frac_pos_right**2 - (1 - frac_pos_right)**2

    quality_scores = -(left_group_sizes / n_objects) * impurity_left - (right_group_sizes / n_objects) * impurity_right

    quality_scores[~valid_partitions] = -np.inf

    optimal_index = np.argmax(quality_scores)
    best_threshold = candidate_thresholds[optimal_index]
    best_score = quality_scores[optimal_index]

    return candidate_thresholds, quality_scores, best_threshold, best_score


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, f"_{key}", value)
        return self

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                positive_indices = np.where(sub_y == 1)[0]
                if len(positive_indices) > 0:
                    clicks = Counter(sub_X[positive_indices, feature])
                else:
                    clicks = Counter()
                ratio = {}
                for key, current_count in counts.items():
                    current_click = clicks.get(key, 0)
                    ratio[key] = current_click / current_count if current_count > 0 else 0
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map.get(x, -1), sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}

        if self._min_samples_leaf is not None:
            if np.sum(split) < self._min_samples_leaf or np.sum(~split) < self._min_samples_leaf:
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return

        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        if self._feature_types[feature_idx] == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)