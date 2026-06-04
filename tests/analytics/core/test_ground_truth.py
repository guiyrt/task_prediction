from datetime import datetime, timedelta, timezone

from task_prediction.models import TaskLabel, TaskType, RunId, TaskGroundTruth
from task_prediction.processing.helpers.ground_truth import build_ground_truth_boundaries

# --- Mocks / Helpers ---
# We use a fixed base time so our tests are deterministic and easy to read
BASE_TIME = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
RUN_ID = RunId(0, 0)

def t_sec(seconds: int) -> datetime:
    """Helper to generate datetimes exactly `seconds` after BASE_TIME."""
    return BASE_TIME + timedelta(seconds=seconds)

def make_label(start_s: int, end_s: int, task_type: TaskType, callsigns: list[str] = []) -> TaskLabel:
    """Factory to quickly create valid TaskLabels for testing."""
    return TaskLabel(
        participant_id=RUN_ID.participant_id,
        scenario_id=RUN_ID.scenario_id,
        asa_support_mode=None,
        start_time=t_sec(start_s),
        end_time=t_sec(end_s),
        task_type=task_type,
        callsigns=callsigns
    )

def assert_gts(gts: list[TaskGroundTruth], expected_dict: dict[int, tuple[TaskType, list[str]]]):
    """Helper to convert the expected dict (using seconds) to a DataFrame for clean assertion."""
    expected_gts = [
        TaskGroundTruth(
            RUN_ID.participant_id,
            RUN_ID.scenario_id,
            t_sec(sec),
            task,
            callsigns
        )
        for sec, (task, callsigns) in expected_dict.items()
    ]
    
    assert gts == expected_gts

# --- Tests ---

def test_empty_list():
    """An empty list should return an empty DataFrame of object type."""
    result = build_ground_truth_boundaries([], RUN_ID)
    assert_gts(result, {})

def test_single_task():
    """A single task should just record its start boundary."""
    labels = [make_label(0, 10, TaskType.IDLE)]
    result = build_ground_truth_boundaries(labels, RUN_ID)
    
    expected = {
        0: (TaskType.IDLE, []),
        10: (None, [])
    }
    assert_gts(result, expected)


def test_sequential_tasks_no_overlap():
    """Tasks that perfectly follow one another without overlapping."""
    labels = [
        make_label(0, 10, TaskType.IDLE),
        make_label(10, 20, TaskType.ASSUME, ["RN123"]),
        make_label(20, 30, TaskType.TRANSFER, ["RN124"])
    ]
    result = build_ground_truth_boundaries(labels, RUN_ID)
    
    expected = {
        0: (TaskType.IDLE, []),
        10: (TaskType.ASSUME, ["RN123"]),
        20: (TaskType.TRANSFER, ["RN124"]),
        30: (None, [])
    }
    assert_gts(result, expected)


def test_simple_overlap_resumption():
    """
    Task A is interrupted by Task B. 
    When B finishes, Task A resumes.
    """
    labels = [
        make_label(0, 20, TaskType.IDLE),       # Task A
        make_label(5, 15, TaskType.ASSUME, ["RN125"])      # Task B (Interrupts A)
    ]
    result = build_ground_truth_boundaries(labels, RUN_ID)
    
    expected = {
        0: (TaskType.IDLE, []),
        5: (TaskType.ASSUME, ["RN125"]),
        15: (TaskType.IDLE, []),     # B ends, resumes A
        20: (None, [])
    }
    assert_gts(result, expected)


def test_nested_overlaps():
    """
    Inception-style nested interruptions:
    A is interrupted by B. B is interrupted by C.
    C ends -> resumes B. B ends -> resumes A.
    """
    labels = [
        make_label(0, 30, TaskType.IDLE),                 # A
        make_label(5, 25, TaskType.ASSUME, ["RN126"]),               # B
        make_label(10, 20, TaskType.CONFLICT_RESOLUTION, ["RN127"])  # C
    ]
    result = build_ground_truth_boundaries(labels, RUN_ID)
    
    expected = {
        0: (TaskType.IDLE, []),
        5: (TaskType.ASSUME, ["RN126"]),
        10: (TaskType.CONFLICT_RESOLUTION, ["RN127"]),
        20: (TaskType.ASSUME, ["RN126"]),               # C ends, resumes B
        25: (TaskType.IDLE, []),                  # B ends, resumes A
        30: (None, [])
    }
    assert_gts(result, expected)


def test_shadow_expiry():
    """
    Crucial edge case:
    A is interrupted by B. 
    A expires *while* B is still active.
    When B ends, A should NOT resume because it is already dead.
    """
    labels = [
        make_label(0, 10, TaskType.IDLE),       # A (ends at 10)
        make_label(5, 15, TaskType.ASSUME, ["RN128"]),     # B (ends at 15)
        make_label(15, 20, TaskType.IDLE),      # C (ends at 20)
    ]
    result = build_ground_truth_boundaries(labels, RUN_ID)
    
    expected = {
        0: (TaskType.IDLE, []),
        5: (TaskType.ASSUME, ["RN128"]),
        15: (TaskType.IDLE, []),
        20: (None, []),
        # At 15, B ends, but A is already dead. No transition is recorded.
        # (The evaluation loop will naturally ignore predictions > 15s)
    }
    assert_gts(result, expected)


def test_consecutive_same_task_cleanup():
    """
    If an interruption happens, but the new task is exactly the same TaskType 
    as the current one, it shouldn't create redundant boundary markers.
    """
    labels = [
        make_label(0, 20, TaskType.ASSUME, ["RN129"]),     # A
        make_label(5, 10, TaskType.ASSUME, ["RN130"]),     # B (Same task type)
        make_label(20, 30, TaskType.IDLE)       # C
    ]
    result = build_ground_truth_boundaries(labels, RUN_ID)
    
    expected = {
        0: (TaskType.ASSUME, ["RN129"]),
        5: (TaskType.ASSUME, ["RN130"]),
        10: (TaskType.ASSUME, ["RN129"]),
        20: (TaskType.IDLE, []),
        30: (None, []),
    }
    assert_gts(result, expected)