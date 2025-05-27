from enum import Enum
from typing import List, Dict, Optional, NamedTuple


class ActionType(Enum):
    ENTER = 1
    EXIT = 2
    MOVE_OBJ = 3
    INIT = 4


class Event(NamedTuple):
    action_type: ActionType
    agent_id: Optional[str] = None
    obj_id: Optional[str] = None
    location: Optional[str] = None
    timestamp: int = -1  # Useful for debugging and ensuring order

    def __repr__(self):
        if self.action_type == ActionType.INIT:
            return (
                f"INIT(T={self.timestamp}, Obj='{self.obj_id}', Loc='{self.location}')"
            )
        if self.action_type == ActionType.MOVE_OBJ:
            return (
                f"Event(T={self.timestamp}, Agt='{self.agent_id}', "
                f"Act={self.action_type.name}, Obj='{self.obj_id}', Loc='{self.location}')"
            )
        else:
            return (
                f"Event(T={self.timestamp}, Agt='{self.agent_id}', "
                f"Act={self.action_type.name})"
            )

    @classmethod
    def from_symbolic_string(cls, symbolic_string: str, timestamp: int = -1):
        parts = symbolic_string.strip().split()
        if len(parts) < 2:
            raise ValueError("Invalid symbolic string format", symbolic_string)

        if parts[0] == "INIT":
            return cls(
                ActionType.INIT,
                obj_id=parts[1],
                location=parts[2],
                timestamp=timestamp,
            )

        agent_id = parts[0]
        action = parts[1]
        if action == "ENTER":
            return cls(ActionType.ENTER, agent_id, timestamp=timestamp)
        elif action == "EXIT":
            return cls(ActionType.EXIT, agent_id, timestamp=timestamp)
        elif action == "MOVE" and len(parts) >= 4:
            return cls(
                ActionType.MOVE_OBJ,
                agent_id,
                obj_id=parts[2],
                location=parts[3],
                timestamp=timestamp,
            )
        raise ValueError("Invalid symbolic string format", symbolic_string)


class Events(List[Event]):
    def __init__(self, events: List[Event]):
        super().__init__(events)

    @classmethod
    def from_symbolic_string(cls, symbolic_string: str):
        events = [
            Event.from_symbolic_string(event, timestamp=i)
            for i, event in enumerate(symbolic_string.split(";"))
        ]
        return cls(events)

    def __repr__(self):
        return "\n".join(map(str, self))

    def get_initial_location_map(self):
        # set up initial map
        initial_obj_location_map = {}
        for event in self:
            if event.action_type == ActionType.INIT:
                initial_obj_location_map[event.obj_id] = event.location

        return initial_obj_location_map


AGENT_POOL = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    # "K",
    # "L",
    # "M",
    # "N",
    # "O",
    # "P",
    # "Q",
    # "R",
    # "S",
    # "T",
]
OBJECT_POOL = ["obj1", "obj2", "obj3"]
LOCATION_POOL = [
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "L7",
    "L8",
    "L9",
    "L10",
    "L11",
    "L12",
    "L13",
    "L14",
    "L15",
    "L16",
    "L17",
    "L18",
    "L19",
    "L20",
]
