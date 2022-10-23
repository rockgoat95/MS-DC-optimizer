from typing import Callable

def linear_schedule(initial_value) -> Callable[[float], float]:
            def func(progress_remaining: float) -> float:
                return initial_value * (progress_remaining * 0.95 + 0.05)

            return func

def descrete_schedule(epi_num) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        if (epi_num * (1 - progress_remaining)) < 500:
            return 4e-4
        elif (epi_num * (1 - progress_remaining)) < 1000:
            return 2e-4
        else:
            return 1e-4

    return func

def descrete_schedule2(epi_num) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        if (epi_num * (1 - progress_remaining)) < 1000:
            return 4e-4
        elif (epi_num * (1 - progress_remaining)) < 2000:
            return 2e-4
        else:
            return 1e-4

    return func
