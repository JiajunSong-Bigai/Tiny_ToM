import unittest
from utils import (
    get_ground_truth,
    resolve_tom_belief,
    get_agent_presence_at_timestamp,
)
from entities import Events


class TestToM(unittest.TestCase):

    def test_scenario_1_ground_truth(self):
        events = "A ENTER; B ENTER; C ENTER; INIT obj1 L0; A MOVE obj1 L1; A EXIT; B MOVE obj1 L2; B EXIT; C MOVE obj1 L3; C EXIT"
        event_log = Events.from_symbolic_string(events)
        self.assertEqual(get_ground_truth("obj1", event_log), "L3")

    def test_scenario_1_A_thinks_B_thinks_C_thinks(self):
        events = "A ENTER; B ENTER; C ENTER; INIT obj1 L0; A MOVE obj1 L1; A EXIT; B MOVE obj1 L2; B EXIT; C MOVE obj1 L3; C EXIT"
        event_log = Events.from_symbolic_string(events)
        # Expected: A thinks B thinks C thinks obj1 is at L1.
        self.assertEqual(
            resolve_tom_belief(["A", "B", "C"], "obj1", event_log, event_log),
            "L1",
        )

    def test_scenario_2_ground_truth(self):
        events = "A ENTER; B ENTER; C ENTER; D ENTER; E ENTER; INIT obj1 L0; A MOVE obj1 L1; A EXIT; B MOVE obj1 L2; B EXIT; C MOVE obj1 L3; C EXIT; D MOVE obj1 L4; D EXIT; E MOVE obj1 L5; E EXIT"
        event_log = Events.from_symbolic_string(events)
        self.assertEqual(get_ground_truth("obj1", event_log), "L5")

    def test_scenario_2_E_thinks_B_thinks_E_thinks(self):
        events = "A ENTER; B ENTER; C ENTER; D ENTER; E ENTER; INIT obj1 L0; A MOVE obj1 L1; A EXIT; B MOVE obj1 L2; B EXIT; C MOVE obj1 L3; C EXIT; D MOVE obj1 L4; D EXIT; E MOVE obj1 L5; E EXIT"
        event_log = Events.from_symbolic_string(events)
        self.assertEqual(
            resolve_tom_belief(["E", "B", "E"], "obj1", event_log, event_log),
            "L2",
        )

    def test_first_order_beliefs_scenario1(self):
        events = "A ENTER; B ENTER; C ENTER; INIT obj1 L0; A MOVE obj1 L1; A EXIT; B MOVE obj1 L2; B EXIT; C MOVE obj1 L3; C EXIT"
        event_log = Events.from_symbolic_string(events)
        self.assertEqual(
            resolve_tom_belief(["A"], "obj1", event_log, event_log),
            "L1",
        )
        self.assertEqual(
            resolve_tom_belief(["B"], "obj1", event_log, event_log),
            "L2",
        )
        self.assertEqual(
            resolve_tom_belief(["C"], "obj1", event_log, event_log),
            "L3",
        )

    def test_agent_never_enters_or_in_log(self):
        events = "A ENTER; INIT obj1 L0; A MOVE obj1 L1; A EXIT"
        event_log = Events.from_symbolic_string(events)
        self.assertEqual(
            resolve_tom_belief(["F"], "obj1", event_log, event_log),
            "L0",
        )

    def test_object_never_moved(self):
        events = "A ENTER; INIT obj1 L0; A EXIT"
        event_log = Events.from_symbolic_string(events)
        self.assertEqual(get_ground_truth("obj1", event_log), "L0")
        self.assertEqual(
            resolve_tom_belief(["A"], "obj1", event_log, event_log),
            "L0",
        )

    def test_sally_anne_equivalent(self):
        events = "Sally ENTER; Anne ENTER; INIT ball Basket_Initial; Sally MOVE ball Basket; Sally EXIT; Anne MOVE ball Box; Anne EXIT"
        full_event_log = Events.from_symbolic_string(events)
        self.assertEqual(get_ground_truth("ball", full_event_log), "Box")
        self.assertEqual(
            resolve_tom_belief(["Sally"], "ball", full_event_log, full_event_log),
            "Basket",
        )
        self.assertEqual(
            resolve_tom_belief(["Anne"], "ball", full_event_log, full_event_log),
            "Box",
        )
        self.assertEqual(
            resolve_tom_belief(
                ["Anne", "Sally"], "ball", full_event_log, full_event_log
            ),
            "Basket",
        )

    def test_sally_anne_sally_returns_sees_nothing_new(self):
        events = "Sally ENTER; Anne ENTER; INIT ball Basket_Initial; Sally MOVE ball Basket; Sally EXIT; Anne MOVE ball Box; Anne EXIT; Sally ENTER"
        full_event_log = Events.from_symbolic_string(events)
        self.assertEqual(
            resolve_tom_belief(["Sally"], "ball", full_event_log, full_event_log),
            "Basket",
        )
        self.assertEqual(
            resolve_tom_belief(["Sally"], "ball", full_event_log, full_event_log),
            "Basket",
        )

    def test_sally_anne_sally_returns_sees_new_move(self):
        events = "Sally ENTER; Anne ENTER; INIT ball Basket_Initial; Sally MOVE ball Basket; Sally EXIT; Anne MOVE ball Box; Anne EXIT; Sally ENTER; Anne MOVE ball Cupboard; Anne EXIT"
        full_event_log = Events.from_symbolic_string(events)
        self.assertEqual(
            resolve_tom_belief(["Sally"], "ball", full_event_log, full_event_log),
            "Cupboard",
        )

    def test_scenario_basic_first_order_A_sees_move(self):
        events_str = "A ENTER; INIT box Table; A MOVE box Shelf; A EXIT"
        event_log = Events.from_symbolic_string(events_str)
        # Expected: A thinks box is at Shelf.
        self.assertEqual(
            resolve_tom_belief(["A"], "box", event_log, event_log),
            "Shelf",
        )

    def test_scenario_A_thinks_B_thinks_A_thinks_after_A_leaves(self):
        events_str = "A ENTER; B ENTER; INIT key Hook; A MOVE key Drawer; A EXIT; B MOVE key Safe; B EXIT"
        event_log = Events.from_symbolic_string(events_str)
        # A thinks key is in Drawer.
        # A thinks B saw A put key in Drawer.
        # A thinks B thinks A (still) thinks key is in Drawer.
        # Expected: A thinks B thinks A thinks key is at Drawer.
        self.assertEqual(
            resolve_tom_belief(["A", "B", "A"], "key", event_log, event_log),
            "Drawer",
        )

    def test_scenario_C_observes_A_then_B_then_A_leaves(self):
        events_str = "A ENTER; B ENTER; C ENTER; INIT book Shelf; A MOVE book Table; B MOVE book Chair; A EXIT; C EXIT; B EXIT"
        event_log = Events.from_symbolic_string(events_str)
        # C sees A move to Table, then B move to Chair. A leaves. C leaves.
        # What does C think A thinks? C saw A move to Table. A left before B's move.
        # So C thinks A thinks it's at Table.
        # Expected: C thinks A thinks book is at Table.
        self.assertEqual(
            resolve_tom_belief(["C", "A"], "book", event_log, event_log),
            "Table",
        )
        # What does C think B thinks? C saw B move to Chair.
        # Expected: C thinks B thinks book is at Chair.
        self.assertEqual(
            resolve_tom_belief(["C", "B"], "book", event_log, event_log),
            "Chair",
        )

    def test_scenario_agent_re_enters_misses_interim_move(self):
        events_str = "A ENTER; B ENTER; INIT cup Counter; A MOVE cup Sink; A EXIT; B MOVE cup Dishwasher; B EXIT; A ENTER; A EXIT"
        event_log = Events.from_symbolic_string(events_str)
        # A puts cup in Sink, exits. B moves cup to Dishwasher. A re-enters but sees no new move.
        # Expected: A thinks cup is at Sink.
        self.assertEqual(
            resolve_tom_belief(["A"], "cup", event_log, event_log),
            "Sink",
        )

    def test_scenario_agent_re_enters_sees_new_move(self):
        events_str = "A ENTER; B ENTER; INIT cup Counter; A MOVE cup Sink; A EXIT; B MOVE cup Dishwasher; A ENTER; B MOVE cup Cabinet; B EXIT; A EXIT"
        event_log = Events.from_symbolic_string(events_str)
        # A puts cup in Sink, exits. B moves to Dishwasher (A misses). A re-enters. B moves to Cabinet (A sees).
        # Expected: A thinks cup is at Cabinet.
        self.assertEqual(
            resolve_tom_belief(["A"], "cup", event_log, event_log),
            "Cabinet",
        )

    def test_scenario_higher_order_missed_information_chain(self):
        events_str = "H ENTER; R ENTER; S ENTER; INIT wand Box; H MOVE wand Shelf; H EXIT; R MOVE wand Cloak; R EXIT; S MOVE wand Pocket; S EXIT"
        event_log = Events.from_symbolic_string(events_str)
        # H thinks wand is at Shelf.
        # H thinks R saw H put wand on Shelf. So H thinks R thinks wand is on Shelf.
        # H thinks R knows S was there.
        # H thinks R thinks S saw H put wand on Shelf.
        # So H thinks R thinks S thinks wand is at Shelf.
        # Expected: H thinks R thinks S thinks wand is at Shelf.
        self.assertEqual(
            resolve_tom_belief(["H", "R", "S"], "wand", event_log, event_log),
            "Shelf",
        )
        # What does R think S thinks?
        # R put wand in Cloak. R saw S was there *before* R moved it from Shelf to Cloak.
        # R knows S saw H put wand on Shelf. R moved it to Cloak. R doesn't know if S saw R's move if S was not looking/left before R's move.
        # Let's assume S sees R's move.
        # R moved to Cloak. S saw H->Shelf, then S saw R->Cloak.
        # So R thinks S thinks it is at Cloak.
        # If S left before R's move: R thinks S thinks Shelf.
        # Let's use the original event order: H->Shelf (S sees). H exits. R->Cloak (S sees). R exits. S->Pocket.
        # R's perspective: H->Shelf. R->Cloak. S knows this.
        # Expected: R thinks S thinks wand is at Cloak.
        self.assertEqual(
            resolve_tom_belief(["R", "S"], "wand", event_log, event_log),
            "Cloak",
        )

    def test_scenario_no_one_moves_object(self):
        events_str = "A ENTER; B ENTER; INIT stone Pedestal; A EXIT; B EXIT"
        event_log = Events.from_symbolic_string(events_str)
        # Expected: A thinks stone is at Pedestal.
        self.assertEqual(
            resolve_tom_belief(["A"], "stone", event_log, event_log),
            "Pedestal",
        )
        # Expected: A thinks B thinks stone is at Pedestal.
        self.assertEqual(
            resolve_tom_belief(["A", "B"], "stone", event_log, event_log),
            "Pedestal",
        )

    def test_scenario_agent_X_never_enters(self):
        events_str = "A ENTER; INIT orb Vault; A MOVE orb Desk; A EXIT"
        event_log = Events.from_symbolic_string(events_str)
        # Agent X is not in the log.
        # Expected: X thinks orb is at Vault (initial state).
        self.assertEqual(
            resolve_tom_belief(["X"], "orb", event_log, event_log),
            "Vault",
        )
        # Expected: A thinks X thinks orb is at Vault.
        self.assertEqual(
            resolve_tom_belief(["A", "X"], "orb", event_log, event_log),
            "Vault",
        )

    def test_scenario_E_thinks_D_thinks_C_thinks_B_thinks_A_thinks(self):
        events_str = (
            "A ENTER; B ENTER; C ENTER; D ENTER; E ENTER; "
            "A MOVE obj1 L1; A EXIT; "
            "B MOVE obj1 L2; B EXIT; "
            "C MOVE obj1 L3; C EXIT; "
            "D MOVE obj1 L4; D EXIT; "
            "E MOVE obj1 L5; E EXIT"
        )
        event_log = Events.from_symbolic_string(events_str)
        # E knows everything up to L5.
        # E knows D left when obj was at L4.
        # E knows D thinks C left when obj was at L3.
        # E knows D thinks C thinks B left when obj was at L2.
        # E knows D thinks C thinks B thinks A left when obj was at L1.
        # E knows D thinks C thinks B thinks A thinks obj1 is at L1.
        # So, what E thinks D thinks C thinks B thinks A thinks obj1 is at L1
        self.assertEqual(
            resolve_tom_belief(["E", "D", "C", "B", "A"], "obj1", event_log, event_log),
            "L1",
        )

    def test_scenario_A_thinks_C_thinks_B_thinks_obj(self):
        events_str = (
            "A ENTER; B ENTER; C ENTER; "
            "INIT item Start; "
            "A MOVE item LocA1; "  # A,B,C see this
            "B MOVE item LocB1; "  # A,C see this
            "A EXIT; "  # B,C see A leave
            "C MOVE item LocC1; "  # B sees this
            "B EXIT; "  # C sees B leave
            "C EXIT"
        )
        event_log = Events.from_symbolic_string(events_str)
        # A's perspective: A->LocA1, B->LocB1. A exits. A does not see C->LocC1.
        # A thinks item is at LocB1.

        # What A thinks C thinks:
        # A knows C saw A->LocA1.
        # A knows C saw B->LocB1.
        # A knows C saw A leave.
        # A does not know about C->LocC1.
        # So A thinks C's last info about item is LocB1 (from B's move which C saw).
        # So A thinks C thinks item is at LocB1.

        # What A thinks C thinks B thinks:
        # A knows C saw B move to LocB1.
        # A knows C saw B see A->LocA1.
        # A knows C saw B see B->LocB1 (B's own move).
        # A knows C saw A leave.
        # A knows C would assume B also saw A leave.
        # From C's perspective (as A models it): B's last known location for the item is LocB1.
        # So, A thinks C thinks B thinks item is at LocB1.
        self.assertEqual(
            resolve_tom_belief(["A", "C", "B"], "item", event_log, event_log),
            "LocB1",
        )

    # Test for the new helper function
    def test_get_agent_presence_at_timestamp(self):
        events = "X ENTER; Y ENTER; X EXIT; Y MOVE o L; X ENTER"
        log = Events.from_symbolic_string(events)

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
