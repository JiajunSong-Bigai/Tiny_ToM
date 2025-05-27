import unittest
from entities import ActionType
from utils import _prepare_event_log_from_dicts
from utils import *


class TestToM(unittest.TestCase):

    def test_scenario_1_ground_truth(self):
        initial_map = {"obj1": "L_initial"}
        raw_events = [
            {"agent_id": "A", "action_type": ActionType.ENTER},
            {"agent_id": "B", "action_type": ActionType.ENTER},
            {"agent_id": "C", "action_type": ActionType.ENTER},
            {
                "agent_id": "A",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L1",
            },
            {"agent_id": "A", "action_type": ActionType.EXIT},
            {
                "agent_id": "B",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L2",
            },
            {"agent_id": "B", "action_type": ActionType.EXIT},
            {
                "agent_id": "C",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L3",
            },
            {"agent_id": "C", "action_type": ActionType.EXIT},
        ]
        event_log = _prepare_event_log_from_dicts(raw_events)
        self.assertEqual(get_ground_truth("obj1", event_log, initial_map), "L3")

    def test_scenario_1_A_thinks_B_thinks_C_thinks(self):
        initial_map = {"obj1": "L_initial"}
        raw_events = [
            {"agent_id": "A", "action_type": ActionType.ENTER},  # 0
            {"agent_id": "B", "action_type": ActionType.ENTER},  # 1
            {"agent_id": "C", "action_type": ActionType.ENTER},  # 2
            {
                "agent_id": "A",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L1",
            },  # 3
            {"agent_id": "A", "action_type": ActionType.EXIT},  # 4
            {
                "agent_id": "B",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L2",
            },  # 5
            {"agent_id": "B", "action_type": ActionType.EXIT},  # 6
            {
                "agent_id": "C",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L3",
            },  # 7
            {"agent_id": "C", "action_type": ActionType.EXIT},  # 8
        ]
        full_event_log = _prepare_event_log_from_dicts(raw_events)
        # Expected: A thinks B thinks C thinks obj1 is at L1.
        self.assertEqual(
            resolve_tom_belief(
                ["A", "B", "C"], "obj1", full_event_log, initial_map, full_event_log
            ),
            "L1",
        )

    def test_scenario_2_ground_truth(self):
        initial_map = {"obj1": "L_initial"}
        raw_events = [
            {"agent_id": "A", "action_type": ActionType.ENTER},
            {"agent_id": "B", "action_type": ActionType.ENTER},
            {"agent_id": "C", "action_type": ActionType.ENTER},
            {"agent_id": "D", "action_type": ActionType.ENTER},
            {"agent_id": "E", "action_type": ActionType.ENTER},
            {
                "agent_id": "A",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L1",
            },
            {"agent_id": "A", "action_type": ActionType.EXIT},
            {
                "agent_id": "B",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L2",
            },
            {"agent_id": "B", "action_type": ActionType.EXIT},
            {
                "agent_id": "C",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L3",
            },
            {"agent_id": "C", "action_type": ActionType.EXIT},
            {
                "agent_id": "D",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L4",
            },
            {"agent_id": "D", "action_type": ActionType.EXIT},
            {
                "agent_id": "E",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L5",
            },
            {"agent_id": "E", "action_type": ActionType.EXIT},
        ]
        event_log = _prepare_event_log_from_dicts(raw_events)
        self.assertEqual(get_ground_truth("obj1", event_log, initial_map), "L5")

    def test_scenario_2_E_thinks_B_thinks_E_thinks(self):
        # Expected: E thinks B thinks E thinks obj1 is at L2.
        initial_map = {"obj1": "L_initial"}
        raw_events = [
            {"agent_id": "A", "action_type": ActionType.ENTER},  # 0
            {"agent_id": "B", "action_type": ActionType.ENTER},  # 1
            {"agent_id": "C", "action_type": ActionType.ENTER},  # 2
            {"agent_id": "D", "action_type": ActionType.ENTER},  # 3
            {"agent_id": "E", "action_type": ActionType.ENTER},  # 4
            {
                "agent_id": "A",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L1",
            },  # 5
            {"agent_id": "A", "action_type": ActionType.EXIT},  # 6
            {
                "agent_id": "B",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L2",
            },  # 7
            {"agent_id": "B", "action_type": ActionType.EXIT},  # 8
            {
                "agent_id": "C",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L3",
            },  # 9
            {"agent_id": "C", "action_type": ActionType.EXIT},  # 10
            {
                "agent_id": "D",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L4",
            },  # 11
            {"agent_id": "D", "action_type": ActionType.EXIT},  # 12
            {
                "agent_id": "E",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L5",
            },  # 13
            {"agent_id": "E", "action_type": ActionType.EXIT},  # 14
        ]
        full_event_log = _prepare_event_log_from_dicts(raw_events)
        self.assertEqual(
            resolve_tom_belief(
                ["E", "B", "E"], "obj1", full_event_log, initial_map, full_event_log
            ),
            "L2",
        )

    def test_first_order_beliefs_scenario1(self):
        initial_map = {"obj1": "L_initial"}
        raw_events = [
            {"agent_id": "A", "action_type": ActionType.ENTER},
            {"agent_id": "B", "action_type": ActionType.ENTER},
            {"agent_id": "C", "action_type": ActionType.ENTER},
            {
                "agent_id": "A",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L1",
            },
            {"agent_id": "A", "action_type": ActionType.EXIT},
            {
                "agent_id": "B",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L2",
            },
            {"agent_id": "B", "action_type": ActionType.EXIT},
            {
                "agent_id": "C",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L3",
            },
            {"agent_id": "C", "action_type": ActionType.EXIT},
        ]
        full_event_log = _prepare_event_log_from_dicts(raw_events)

        self.assertEqual(
            resolve_tom_belief(
                ["A"], "obj1", full_event_log, initial_map, full_event_log
            ),
            "L1",
        )
        self.assertEqual(
            resolve_tom_belief(
                ["B"], "obj1", full_event_log, initial_map, full_event_log
            ),
            "L2",
        )
        self.assertEqual(
            resolve_tom_belief(
                ["C"], "obj1", full_event_log, initial_map, full_event_log
            ),
            "L3",
        )

    def test_agent_never_enters_or_in_log(self):
        initial_map = {"obj1": "L_initial"}
        raw_events = [
            {"agent_id": "A", "action_type": ActionType.ENTER},
            {
                "agent_id": "A",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "obj1",
                "location": "L1",
            },
            {"agent_id": "A", "action_type": ActionType.EXIT},
        ]
        full_event_log = _prepare_event_log_from_dicts(raw_events)
        self.assertEqual(
            resolve_tom_belief(
                ["F"], "obj1", full_event_log, initial_map, full_event_log
            ),
            "L_initial",
        )

    def test_object_never_moved(self):
        initial_map = {"obj1": "L_initial"}
        raw_events = [
            {"agent_id": "A", "action_type": ActionType.ENTER},
            {"agent_id": "A", "action_type": ActionType.EXIT},
        ]
        full_event_log = _prepare_event_log_from_dicts(raw_events)
        self.assertEqual(
            get_ground_truth("obj1", full_event_log, initial_map), "L_initial"
        )
        self.assertEqual(
            resolve_tom_belief(
                ["A"], "obj1", full_event_log, initial_map, full_event_log
            ),
            "L_initial",
        )

    def test_sally_anne_equivalent(self):
        initial_map = {"ball": "Basket_Initial"}
        raw_events = [
            {"agent_id": "Sally", "action_type": ActionType.ENTER},
            {"agent_id": "Anne", "action_type": ActionType.ENTER},
            {
                "agent_id": "Sally",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "ball",
                "location": "Basket",
            },
            {"agent_id": "Sally", "action_type": ActionType.EXIT},
            {
                "agent_id": "Anne",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "ball",
                "location": "Box",
            },
            {"agent_id": "Anne", "action_type": ActionType.EXIT},
        ]
        full_event_log = _prepare_event_log_from_dicts(raw_events)

        self.assertEqual(get_ground_truth("ball", full_event_log, initial_map), "Box")
        self.assertEqual(
            resolve_tom_belief(
                ["Sally"], "ball", full_event_log, initial_map, full_event_log
            ),
            "Basket",
        )
        self.assertEqual(
            resolve_tom_belief(
                ["Anne"], "ball", full_event_log, initial_map, full_event_log
            ),
            "Box",
        )
        self.assertEqual(
            resolve_tom_belief(
                ["Anne", "Sally"], "ball", full_event_log, initial_map, full_event_log
            ),
            "Basket",
        )

    def test_sally_anne_sally_returns_sees_nothing_new(self):
        initial_map = {"ball": "L_initial"}
        raw_events = [
            {"agent_id": "Sally", "action_type": ActionType.ENTER},
            {"agent_id": "Anne", "action_type": ActionType.ENTER},
            {
                "agent_id": "Sally",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "ball",
                "location": "Basket",
            },
            {"agent_id": "Sally", "action_type": ActionType.EXIT},
            {
                "agent_id": "Anne",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "ball",
                "location": "Box",
            },
            {"agent_id": "Sally", "action_type": ActionType.ENTER},
        ]
        full_event_log = _prepare_event_log_from_dicts(raw_events)
        self.assertEqual(
            resolve_tom_belief(
                ["Sally"], "ball", full_event_log, initial_map, full_event_log
            ),
            "Basket",
        )

    def test_sally_anne_sally_returns_sees_new_move(self):
        initial_map = {"ball": "L_initial"}
        raw_events = [
            {"agent_id": "Sally", "action_type": ActionType.ENTER},
            {"agent_id": "Anne", "action_type": ActionType.ENTER},
            {
                "agent_id": "Sally",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "ball",
                "location": "Basket",
            },
            {"agent_id": "Sally", "action_type": ActionType.EXIT},
            {
                "agent_id": "Anne",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "ball",
                "location": "Box",
            },
            {"agent_id": "Sally", "action_type": ActionType.ENTER},
            {
                "agent_id": "Anne",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "ball",
                "location": "Cupboard",
            },
        ]
        full_event_log = _prepare_event_log_from_dicts(raw_events)
        self.assertEqual(
            resolve_tom_belief(
                ["Sally"], "ball", full_event_log, initial_map, full_event_log
            ),
            "Cupboard",
        )

    # Test for the new helper function
    def test_get_agent_presence_at_timestamp(self):
        raw_events_presence = [
            {"agent_id": "X", "action_type": ActionType.ENTER},  # ts 0
            {"agent_id": "Y", "action_type": ActionType.ENTER},  # ts 1
            {"agent_id": "X", "action_type": ActionType.EXIT},  # ts 2
            {
                "agent_id": "Y",
                "action_type": ActionType.MOVE_OBJ,
                "obj_id": "o",
                "location": "L",
            },  # ts 3
            {"agent_id": "X", "action_type": ActionType.ENTER},  # ts 4
        ]
        log = _prepare_event_log_from_dicts(raw_events_presence)

        self.assertFalse(
            get_agent_presence_at_timestamp("X", 0, log)
        )  # Before X's first entry
        self.assertTrue(get_agent_presence_at_timestamp("X", 1, log))  # X entered at 0
        self.assertTrue(
            get_agent_presence_at_timestamp("X", 2, log)
        )  # X still in from ts 0, exit is at ts 2
        self.assertFalse(get_agent_presence_at_timestamp("X", 3, log))  # X exited at 2
        self.assertFalse(
            get_agent_presence_at_timestamp("X", 4, log)
        )  # X still out from ts 2, re-entry is at ts 4
        self.assertTrue(
            get_agent_presence_at_timestamp("X", 5, log)
        )  # X re-entered at 4

        self.assertFalse(
            get_agent_presence_at_timestamp("Y", 1, log)
        )  # Before Y's first entry
        self.assertTrue(get_agent_presence_at_timestamp("Y", 2, log))  # Y entered at 1
        self.assertTrue(
            get_agent_presence_at_timestamp("Y", 10, log)
        )  # Y entered at 1 and never exited

        self.assertFalse(get_agent_presence_at_timestamp("Z", 5, log))  # Z never in log


if __name__ == "__main__":
    unittest.main()
