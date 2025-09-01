---
name: plan-mode-runner
description: Multi-step planner and executor that supports checkpoints, resumable plans, and recovery
tools:
  - plan
  - execute
  - checkpoint
  - resume
  - replan
  - log
model: sonnet
---

from datetime import datetime
import uuid

class PlanState:
    def __init__(self):
        self.plan_id = str(uuid.uuid4())
        self.steps = []
        self.current_step = 0
        self.status = "initialized"
        self.history = []

    def add_step(self, step_description):
        self.steps.append({
            "step_id": len(self.steps) + 1,
            "description": step_description,
            "status": "pending",
            "started_at": None,
            "ended_at": None,
            "logs": []
        })

    def mark_step_started(self, index):
        self.steps[index]["status"] = "in_progress"
        self.steps[index]["started_at"] = datetime.utcnow().isoformat()

    def mark_step_done(self, index):
        self.steps[index]["status"] = "done"
        self.steps[index]["ended_at"] = datetime.utcnow().isoformat()

    def mark_step_failed(self, index, error_msg):
        self.steps[index]["status"] = "failed"
        self.steps[index]["ended_at"] = datetime.utcnow().isoformat()
        self.steps[index]["logs"].append(f"ERROR: {error_msg}")
        self.status = "failed"

    def get_next_step(self):
        for i, step in enumerate(self.steps):
            if step["status"] == "pending":
                self.current_step = i
                return step
        return None

    def serialize(self):
        return {
            "plan_id": self.plan_id,
            "status": self.status,
            "steps": self.steps,
            "history": self.history
        }

##  Plan + Execute Lifecycle
def run_plan(plan_steps):
    plan = PlanState()
    for step in plan_steps:
        plan.add_step(step)

    while True:
        step = plan.get_next_step()
        if step is None:
            break
        try:
            plan.mark_step_started(plan.current_step)
            log(f"Executing Step {plan.current_step + 1}: {step['description']}")
            result = execute_step(step["description"])
            validate_result(result)
            plan.mark_step_done(plan.current_step)
        except Exception as e:
            plan.mark_step_failed(plan.current_step, str(e))
            save_checkpoint(plan)
            notify("Execution paused due to error.")
            break

    return plan.serialize()
## Checkpoint + Resume Logic
def save_checkpoint(plan):
    with open(f"./checkpoints/{plan.plan_id}.json", "w") as f:
        f.write(json.dumps(plan.serialize(), indent=2))

def resume_plan(plan_id):
    with open(f"./checkpoints/{plan_id}.json") as f:
        data = json.load(f)
    plan = PlanState()
    plan.plan_id = data["plan_id"]
    plan.steps = data["steps"]
    plan.status = "resumed"
    return plan
