
#!/usr/bin/env python3

import os
import json
import uuid
import argparse
from datetime import datetime

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class PlanState:
    def __init__(self, plan_id=None):
        self.plan_id = plan_id or str(uuid.uuid4())
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

    def save_checkpoint(self):
        with open(f"{CHECKPOINT_DIR}/{self.plan_id}.json", "w") as f:
            json.dump(self.serialize(), f, indent=2)

    @classmethod
    def load_checkpoint(cls, plan_id):
        with open(f"{CHECKPOINT_DIR}/{plan_id}.json") as f:
            data = json.load(f)
        plan = cls(plan_id=data["plan_id"])
        plan.steps = data["steps"]
        plan.status = "resumed"
        return plan

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
            print(f"Executing Step {plan.current_step + 1}: {step['description']}")
            # Simulate success
            plan.mark_step_done(plan.current_step)
        except Exception as e:
            plan.mark_step_failed(plan.current_step, str(e))
            plan.save_checkpoint()
            print(f"Execution paused due to error: {e}")
            break

    plan.save_checkpoint()
    return plan

def main():
    parser = argparse.ArgumentParser(description="Claude Plan Mode CLI Utility")
    parser.add_argument("--plan", nargs='+', help="List of plan steps to execute")
    parser.add_argument("--resume", type=str, help="Plan ID to resume")

    args = parser.parse_args()

    if args.resume:
        print(f"Resuming plan {args.resume}")
        plan = PlanState.load_checkpoint(args.resume)
        run_plan([step["description"] for step in plan.steps if step["status"] == "pending"])
    elif args.plan:
        print("Starting new plan...")
        plan = run_plan(args.plan)
        print("Plan complete. Plan ID:", plan.plan_id)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
