import numpy as np


class Apriori:
    def __init__(self, min_support: float, min_confidence: float, data: dict) -> None:
        self.__min_support: float = min_support
        self.__min_confidence: float = min_confidence
        self.__data: dict = data
        self.__rules: list = []
        self.__support_set: dict = {}

    def get_rules(self) -> list:
        return self.__rules

    def get_support_set(self) -> dict:
        return self.__support_set

    def __str__(self) -> str:
        return f'Apriori Algorithm:\nMin support: {self.__min_support}\nMin confidence: {self.__min_confidence}'

