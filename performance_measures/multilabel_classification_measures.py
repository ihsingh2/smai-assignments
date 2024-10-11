""" Provides MultiLabelClassificationMeasures. """

import numpy as np
import numpy.typing as npt


class MultiLabelClassificationMeasures:
    """ Computes common evaluation measures for multi-label classification based tasks. """


    def __init__(self, y_true: npt.NDArray, y_pred: npt.NDArray):
        """ Initializes the class to compute on given data.

        Args:
            y_true: Array containing list of true values.
            y_pred: Array containing list of predicted values.
        """

        # Convert the passed arguments to set to ease computation
        self.y_true = np.array([set(labels) for labels in y_true])
        self.y_pred = np.array([set(labels) for labels in y_pred])

        # Initialize the classes based on y values
        y_concat = np.concatenate((y_true, y_pred))
        self.num_classes = max(val for sublist in y_concat for val in sublist) + 1


    def accuracy_score(self) -> float:
        """ Computes the accuracy. """

        return np.mean(np.array([ \
                                    len(t & p) / len(t | p) if len(t | p) > 0 else 1.0 \
                                                    for t, p in zip(self.y_true, self.y_pred)]))


    def f1_score(self) -> float:
        """ Computes the f1 score. """

        # Compute recall and precision
        recall = self.recall_score()
        precision = self.precision_score()

        # Compute the F1 score
        f1 = 2 * recall * precision / (recall + precision)

        return f1


    def hamming_distance(self) -> float:
        """ Computes the hamming distance. """

        return np.mean(np.array([ \
                                    len((t ^ p)) / self.num_classes \
                                                    for t, p in zip(self.y_true, self.y_pred)]))


    def recall_score(self) -> float:
        """ Computes the recall. """

        return np.mean(np.array([ \
                                    len(t & p) / len(t) if len(t) > 0 else 1.0
                                                    for t, p in zip(self.y_true, self.y_pred)]))


    def precision_score(self) -> float:
        """ Computes the precision. """

        return np.mean(np.array([ \
                                    len(t & p) / len(p) if len(p) > 0 else 1.0 \
                                                    for t, p in zip(self.y_true, self.y_pred)]))


    def print_all_measures(self) -> None:
        """ Evaluates and prints all the measures. """

        print('Accuracy:', self.accuracy_score())
        print('F1 Score:', self.f1_score())
        print('Hamming Distance:', self.hamming_distance())
        print('Precision:', self.precision_score())
        print('Recall:', self.recall_score())
