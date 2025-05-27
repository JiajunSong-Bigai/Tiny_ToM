from entities import Event, ActionType, Events
from typing import List, Dict, Optional
import os
import torch.backends.cudnn as cudnn
import torch
import random
import numpy as np


def get_agent_direct_belief(
    agent_id_of_believer: str,
    target_obj_id: str,
    perspective_log: List[Event],  # The world as this agent has perceived it
    initial_obj_location_map: Dict[str, str],
) -> Optional[str]:
    # """
    # Determines what `agent_id_of_believer` directly believes the `target_obj_id`'s
    # location is, based on their chronologically ordered `perspective_log`.
    # """
    # believed_location = initial_obj_location_map.get(target_obj_id)

    # # Tracks the believer's presence in the room *according to their perspective_log*.
    # # This is used to determine if they witnessed events performed by OTHERS in their log.
    # agent_is_in_room_for_belief_calc = False

    # for event in perspective_log:
    #     # 1. Update the believer's presence status based on THEIR OWN actions in this log
    #     if event.agent_id == agent_id_of_believer:
    #         if event.action_type == ActionType.ENTER:
    #             agent_is_in_room_for_belief_calc = True
    #         elif event.action_type == ActionType.EXIT:
    #             agent_is_in_room_for_belief_calc = False

    #     # 2. Update belief about target_obj_id based on relevant MOVE_OBJ events
    #     if event.action_type == ActionType.MOVE_OBJ and event.obj_id == target_obj_id:
    #         if event.agent_id == agent_id_of_believer:
    #             # If the believer themself moved the object
    #             believed_location = event.location
    #         elif agent_is_in_room_for_belief_calc:
    #             # If another agent moved the object AND the believer was present to witness it
    #             believed_location = event.location

    # return believed_location
    """
    Determines what `agent_id_of_believer` directly believes the `target_obj_id`'s
    location is, based on their chronologically ordered `perspective_log`.
    The perspective_log is assumed to contain only events the believer
    performed or witnessed.
    """
    believed_location = initial_obj_location_map.get(target_obj_id)

    # No need to track 'agent_is_in_room_for_belief_calc' here anymore for witnessing others' actions,
    # as the perspective_log is assumed to be pre-filtered by resolve_tom_belief
    # to only include events the agent_id_of_believer actually perceived.

    for event in perspective_log:
        if event.action_type == ActionType.MOVE_OBJ and event.obj_id == target_obj_id:
            # If a move event for the target object is in this agent's perspective_log,
            # it means they either performed it or witnessed it.
            # The last such event in their log determines their belief.
            believed_location = event.location

    return believed_location


def get_agent_presence_at_timestamp(
    agent_id_to_check: str,
    check_timestamp: int,  # The timestamp *at which* we want to know presence (exclusive of events at this exact time)
    reference_log: List[Event],
) -> bool:
    """
    Determines if an agent was present in the room *just before* a given timestamp,
    based on their ENTER/EXIT events in the reference_log.
    """
    is_present = False
    last_relevant_event_ts = -1

    for event in reference_log:
        if event.timestamp >= check_timestamp:
            # We only care about events *before* the check_timestamp
            break

        if event.agent_id == agent_id_to_check:
            if event.action_type == ActionType.ENTER:
                # Only update if this event is later than any previously considered ENTER/EXIT for this agent
                if event.timestamp > last_relevant_event_ts:
                    is_present = True
                    last_relevant_event_ts = event.timestamp
            elif event.action_type == ActionType.EXIT:
                if event.timestamp > last_relevant_event_ts:
                    is_present = False
                    last_relevant_event_ts = event.timestamp
    return is_present


def resolve_tom_belief(
    agent_sequence: List[str],
    target_obj_id: str,
    current_event_log: Events,  # The log from which the current_thinker forms their perspective
    # initial_obj_location_map: Dict[str, str],
    full_event_log_for_reference: Events,  # The original, complete event log
) -> Optional[str]:
    if not agent_sequence:
        raise ValueError("Agent sequence cannot be empty for resolve_tom_belief.")

    # set up initial map
    # initial_obj_location_map = {}
    # for event in full_event_log_for_reference:
    #     if event.action_type == ActionType.INIT:
    #         initial_obj_location_map[event.obj_id] = event.location
    initial_obj_location_map = full_event_log_for_reference.get_initial_location_map()

    current_thinker = agent_sequence[0]
    remaining_agent_sequence = agent_sequence[1:]

    perspective_log_for_current_thinker: List[Event] = []

    is_current_thinker_in_room_to_witness = False  # Default
    if current_event_log:  # Only if there are events to process
        start_timestamp_of_current_log = current_event_log[0].timestamp
        # Determine if current_thinker was already in the room *before* the first event of current_event_log
        is_current_thinker_in_room_to_witness = get_agent_presence_at_timestamp(
            current_thinker,
            start_timestamp_of_current_log,
            full_event_log_for_reference,  # Use the full log for this check
        )

    for event in current_event_log:
        # If the current thinker performed the event, they know about it.
        # Their presence status is also updated by their own ENTER/EXIT.
        if event.agent_id == current_thinker:
            perspective_log_for_current_thinker.append(event)
            if event.action_type == ActionType.ENTER:
                is_current_thinker_in_room_to_witness = True
            elif event.action_type == ActionType.EXIT:
                is_current_thinker_in_room_to_witness = False
        # Else, if another agent performed the event, the current thinker only
        # witnesses it if they are currently in the room.
        elif is_current_thinker_in_room_to_witness:
            perspective_log_for_current_thinker.append(event)

    if not remaining_agent_sequence:
        return get_agent_direct_belief(
            current_thinker,
            target_obj_id,
            perspective_log_for_current_thinker,
            initial_obj_location_map,
        )
    else:
        return resolve_tom_belief(
            remaining_agent_sequence,
            target_obj_id,
            perspective_log_for_current_thinker,
            full_event_log_for_reference,  # Pass the full reference log down
        )


def get_ground_truth(
    target_obj_id: str,
    full_event_log: Events,
    # initial_obj_location_map: Dict[str, str],
) -> Optional[str]:
    """
    Determines the actual final location of the target_obj_id based on the complete, unfiltered event_log.
    """
    initial_obj_location_map = full_event_log.get_initial_location_map()
    actual_location = initial_obj_location_map.get(target_obj_id)
    for event in full_event_log:
        if event.action_type == ActionType.MOVE_OBJ and event.obj_id == target_obj_id:
            actual_location = event.location
    return actual_location


#######################################
######## UTILS FOR TRAINING   #########
#######################################


def create_folder(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)


def fix_random_seed(seed, reproduce=False):
    # cudnn.enabled = True
    # cudnn.benchmark = True

    if reproduce:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ## NOTE: uncomment for CUDA >= 10.2
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        ## NOTE: uncomment for pytorch >= 1.8
        # torch.use_deterministic_algorithms(True)

    # os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)

    return rng


#######################################
######## UTILS FOR TOKENIZATION #######
#######################################

from entities import AGENT_POOL, OBJECT_POOL, LOCATION_POOL


def build_vocab(
    agent_pool: List[str] = AGENT_POOL,
    object_pool: List[str] = OBJECT_POOL,
    location_pool: List[str] = LOCATION_POOL,
):
    vocab = {}

    # Add special tokens
    vocab["<PAD>"] = 0
    vocab[";"] = 1

    # Add action words
    vocab["ENTER"] = 2
    vocab["EXIT"] = 3
    vocab["MOVE"] = 4
    vocab["INIT"] = 5
    vocab["BELIEF"] = 6

    # Add agent identifiers
    for i, agent in enumerate(agent_pool):
        vocab[agent] = i + 7

    # Add object identifiers
    offset = 8 + len(agent_pool)
    for i, obj in enumerate(object_pool):
        vocab[obj] = i + offset

    # Add location identifiers
    offset = 8 + len(agent_pool) + len(object_pool)
    for i, loc in enumerate(location_pool):
        vocab[loc] = i + offset

    return vocab


def c_to_i(
    sentence: str,
    agent_pool: List[str] = AGENT_POOL,
    object_pool: List[str] = OBJECT_POOL,
    location_pool: List[str] = LOCATION_POOL,
) -> List[int]:

    vocab = build_vocab(agent_pool, object_pool, location_pool)

    tokens = []
    for word in sentence.split():
        if word in vocab:
            tokens.append(vocab[word])
        else:
            raise ValueError("Unknown token", word)

    return tokens


def i_to_c(
    tokens: List[int],
    agent_pool: List[str] = AGENT_POOL,
    object_pool: List[str] = OBJECT_POOL,
    location_pool: List[str] = LOCATION_POOL,
) -> str:
    """
    Decode a list of integer tokens back into a sentence.

    Args:
        tokens: List of integer tokens
        agent_pool: List of available agent identifiers
        object_pool: List of available object identifiers
        location_pool: List of available location identifiers

    Returns:
        The decoded sentence as a string
    """
    vocab = build_vocab(agent_pool, object_pool, location_pool)
    reverse_vocab = {v: k for k, v in vocab.items()}

    # Decode the tokens
    if not isinstance(tokens, list):
        return reverse_vocab[tokens]

    words = [reverse_vocab[token] for token in tokens]
    sentence = " ".join(words)

    return sentence
