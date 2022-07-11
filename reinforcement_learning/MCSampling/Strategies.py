import numpy as np


class PercentRule:

    def __init__(self, percentage: float, text: str):
        self._percentage = percentage
        self.text = text

    def pick_apartment(self, apartments: list[float]) -> float:
        n = len(apartments)
        split_point = int(self._percentage * n)
        ap_up_to_split = apartments[:split_point]
        ap_from_split = apartments[split_point:]

        highest_ap_until_split_point = np.max(ap_up_to_split)
        next_ap_that_is_higher = ap_from_split[np.argmax(ap_from_split > highest_ap_until_split_point)]
        return next_ap_that_is_higher


class HigherThanNumberRule:

    def __init__(self, number: float, text: str):
        self._number = number
        self.text = text

    def pick_apartment(self, apartments: list[float]) -> float:
        apartment = apartments[np.argmax(apartments > self._number)]
        return apartment
