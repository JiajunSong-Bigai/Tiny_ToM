import random
import json
from typing import List, Dict, Tuple, Optional
import itertools
import os

from entities import Events, OBJECT_POOL, LOCATION_POOL, AGENT_POOL
from utils import resolve_tom_belief, get_ground_truth


class ScenarioGenerator:
    """
    Generator for creating symbolic ToM scenarios with configurable complexity.

    Main control parameters:
    - num_agents: Number of agents in the scenario
    - max_chain_length: Maximum length of belief chains for questions
    """

    def __init__(
        self,
        num_scenarios: int = 10,
        min_agents: int = 2,
        max_agents: int = 5,
        min_chain_length: int = 1,
        max_chain_length: int = 5,
        num_objects: int = 1,
        move_probability: float = 0.5,
        num_questions_per_scenario: float = None,
        object_pool: List[str] = OBJECT_POOL,
        location_pool: List[str] = LOCATION_POOL,
        agent_pool: List[str] = AGENT_POOL,
        seed: Optional[int] = None,
    ):
        """
        Initialize the scenario generator with configurable parameters.
        """
        self.num_scenarios = num_scenarios
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.min_chain_length = min_chain_length
        self.max_chain_length = max_chain_length
        self.num_objects = num_objects
        self.move_probability = move_probability
        self.num_questions_per_scenario = num_questions_per_scenario

        self.object_pool = object_pool
        self.agent_pool = agent_pool
        self.location_pool = location_pool

        if seed is not None:
            random.seed(seed)

    def generate_scenario(
        self,
    ) -> Tuple[Dict[str, str], str, List[Tuple[List[str], str, str]]]:
        """
        Generate a complete scenario with:
        - Initial object locations
        - Event sequence
        - Questions with expected answers

        Returns:
            Tuple of (initial_map, event_string, questions)
            where questions is a list of (agent_sequence, object_id, expected_answer) tuples
        """
        # Generate agents and objects
        num_agents = random.randint(self.min_agents, self.max_agents)
        agents = random.sample(self.agent_pool, num_agents)
        objects = self.object_pool[
            : self.num_objects
        ]  # random.sample(self.object_pool, self.num_objects)
        locations = self.location_pool[1:]

        # Generate event sequence
        events = []

        # All agents enter
        for agent in agents:
            events.append(f"{agent} ENTER")
        for obj in objects:
            events.append(f"INIT {obj} {self.location_pool[0]}")

        # Each agent moves each object once
        for i, agent in enumerate(agents):
            for obj in objects:
                if random.uniform(0, 1) < self.move_probability:
                    events.append(f"{agent} MOVE {obj} {locations[i % len(locations)]}")
            events.append(f"{agent} EXIT")

        event_string = " ; ".join(events)
        event_log = Events.from_symbolic_string(event_string)

        # Generate questions and answers
        qas = []
        for obj in objects:
            # Generate belief chains of varying lengths
            for chain_length in range(self.min_chain_length, self.max_chain_length + 1):
                # Generate a few random agent sequences of this length
                if chain_length == 0:
                    ground_truth = get_ground_truth(obj, event_log)
                    qas.append((f"BELIEF {obj}", ground_truth))
                    continue

                all_agent_sequence = list(itertools.combinations(agents, chain_length))

                if self.num_questions_per_scenario:
                    random.shuffle(all_agent_sequence)
                    all_agent_sequence = all_agent_sequence[
                        : self.num_questions_per_scenario
                    ]

                for agent_sequence in all_agent_sequence:
                    agent_sequence = list(agent_sequence)
                    random.shuffle(agent_sequence)
                    expected_answer = resolve_tom_belief(
                        agent_sequence, obj, event_log, event_log
                    )
                    qas.append(
                        (f"BELIEF {' '.join(agent_sequence)} {obj}", expected_answer)
                    )

        return event_string, qas

    def generate_dataset(self) -> str:
        result = []
        for _ in range(self.num_scenarios):
            event_string, qas = self.generate_scenario()
            for question, answer in qas:
                result.append(
                    {
                        "events": event_string,
                        "question": question,
                        "answer": answer,
                    }
                )

        return {
            "config": self.__dict__,
            "data": result,
        }

    def save_dataset(self, dataset, filename: str):
        directory = "/".join(filename.split("/")[:-1])
        os.makedirs(directory, exist_ok=True)

        with open(filename, "w") as f:
            json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    generator = ScenarioGenerator(
        num_scenarios=100,
        min_agents=1,
        max_agents=10,
        num_objects=1,
        min_chain_length=1,
        max_chain_length=6,
        move_probability=0.8,
        seed=42,
    )
    dataset = generator.generate_dataset()
    print(len(dataset["data"]))

    generator.save_dataset(dataset, "data/dataset.json")
