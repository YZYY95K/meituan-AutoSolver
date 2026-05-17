from __future__ import annotations
from typing import Type
from .greedy import AllocationAlgorithm, GreedyAllocator, AllocationResult
from .or_tools_solver import ORToolsSolver
from .genetic import GeneticAllocator


class AlgorithmRegistry:
    _algorithms: dict[str, Type[AllocationAlgorithm]] = {}

    @classmethod
    def register(cls, algorithm_class: Type[AllocationAlgorithm]) -> None:
        cls._algorithms[algorithm_class.name] = algorithm_class

    @classmethod
    def get(cls, name: str) -> AllocationAlgorithm:
        if name not in cls._algorithms:
            raise ValueError(
                f"Algorithm '{name}' not found. Available: {list(cls._algorithms.keys())}"
            )
        return cls._algorithms[name]()

    @classmethod
    def list_algorithms(cls) -> list[str]:
        return list(cls._algorithms.keys())

    @classmethod
    def get_descriptions(cls) -> dict[str, str]:
        return {
            "greedy": "贪心算法 - 基于距离和优先级的快速分配，适合实时场景",
            "or_tools": "OR-Tools求解器 - 基于约束规划的精确求解，适合中小规模问题",
            "genetic": "遗传算法 - 基于进化的元启发式求解，适合大规模复杂问题",
        }


AlgorithmRegistry.register(GreedyAllocator)
AlgorithmRegistry.register(ORToolsSolver)
AlgorithmRegistry.register(GeneticAllocator)


def get_algorithm(name: str) -> AllocationAlgorithm:
    return AlgorithmRegistry.get(name)
