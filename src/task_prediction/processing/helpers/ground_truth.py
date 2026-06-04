import logging
from ...models import TaskLabel, TaskGroundTruth, RunId

logger = logging.getLogger(__name__)

def build_ground_truth_boundaries(labels: list[TaskLabel], run: RunId) -> list[TaskGroundTruth]:
    if not labels:
        return []

    stack: list[TaskLabel] = []
    current_label: TaskLabel = labels[0]
    transitions: list[TaskGroundTruth] = [
        TaskGroundTruth(
            run.participant_id,
            run.scenario_id,
            current_label.start_time,
            current_label.task_type,
            current_label.callsigns
        )
    ]

    def add_new_task(gt: TaskGroundTruth) -> bool:
        # Make sure it's a transition to a new task
        if (gt.true_task == transitions[-1].true_task) and (gt.true_callsigns == transitions[-1].true_callsigns):
            return
        
        # If timestamp is same, replace old gt
        if gt.timestamp == transitions[-1].timestamp:
            transitions.pop()
        
        transitions.append(gt)

    def consume_stack() -> bool:
        nonlocal current_label

        while stack and stack[-1].end_time <= current_label.end_time:
            stack.pop()
            
        if stack:
            popped = stack.pop()
            add_new_task(
                TaskGroundTruth(
                    run.participant_id,
                    run.scenario_id,
                    current_label.end_time,
                    popped.task_type,
                    popped.callsigns
                )
            )
            current_label = popped
            return True
        
        return False

    for next_label in labels[1:]:
        # 1. Resolve tasks in the stack that resume BEFORE next_label starts
        while next_label.start_time >= current_label.end_time:
            # Purge expired tasks
            if not consume_stack():
                break
                
        # 2. Process next_label (Interrupts current, or follows immediately)
        if next_label.start_time < current_label.end_time:
            stack.append(current_label)
            
        add_new_task(
            TaskGroundTruth(
                run.participant_id,
                run.scenario_id,
                next_label.start_time,
                next_label.task_type,
                next_label.callsigns
            )
        )
        current_label = next_label

    # 3. Final Flush: Resume any remaining tasks in the stack
    while stack:
        consume_stack()
    
    # Mark end of scenario
    add_new_task(
        TaskGroundTruth(
            run.participant_id,
            run.scenario_id,
            current_label.end_time,
            None,
            []
        )
    )

    return transitions