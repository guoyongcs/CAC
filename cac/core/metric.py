import torch


class Accuracy(object):
    def __init__(self, rate, n_correct, n_total):
        self.rate = rate
        self.n_correct = n_correct
        self.n_total = n_total

    def __str__(self):
        return f"Accuracy={self.rate * 100:.4f}%({self.n_correct}/{self.n_total})"


class AccuracyMetric(object):
    def __init__(self, topk=(1,)):
        self.topk = sorted(list(topk))
        self._last_accuracies = None
        self._accuracies = None
        self.reset()

    def reset(self, *args, **kwargs):
        self._accuracies = [
            Accuracy(rate=0.0, n_correct=0, n_total=0) for _ in self.topk]
        self.reset_last()

    def reset_last(self):
        self._last_accuracies = [Accuracy(rate=0.0, n_correct=0, n_total=0) for _ in self.topk]

    def update(self, targets, outputs):
        with torch.no_grad():
            maxk = max(self.topk)
            batch_size = targets.size(0)

            _, pred = outputs.topk(k=maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1))

            for accuracy, last_accuracy, k in zip(self._accuracies, self._last_accuracies, self.topk):
                accuracy.n_total += batch_size

                correct_k = correct[:k].sum().item()
                accuracy.n_correct += correct_k
                accuracy.rate = accuracy.n_correct / accuracy.n_total

                last_accuracy.n_total = batch_size
                last_accuracy.n_correct = correct_k
                last_accuracy.rate = last_accuracy.n_correct / last_accuracy.n_total

    @property
    def value(self):
        return self

    def last_accuracy(self, i) -> Accuracy:
        return self._last_accuracies[self.topk.index(i)]

    def accuracy(self, i) -> Accuracy:
        return self._accuracies[self.topk.index(i)]


class AverageMetric(object):
    def __init__(self):
        self.n = 0
        self._value = 0.
        self.last = 0.

    def reset(self) -> None:
        self.n = 0
        self._value = 0.
        self.last = 0.

    def update(self, value) -> None:
        if torch.is_tensor(value):
            self.last = value.item()
        elif isinstance(value, (int, float)):
            self.last = value
        else:
            raise ValueError(f"The parameter 'value' should be int, float or pytorch scalar tensor, "
                             f"but found {type(value)}")
        self.n += 1
        self._value += self.last

    @property
    def value(self) -> float:
        if self.n == 0:
            raise ValueError("The container is empty.")
        return self._value / self.n
