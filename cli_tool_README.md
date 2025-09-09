Here’s a complete README.md for your Plan Runner CLI utility:

⸻

# 🧠 Plan Runner CLI

**Plan Runner CLI** is a lightweight command-line tool for executing and resuming structured
multi-step workflows (called _plans_) with checkpointing and context tracking.

Ideal for AI-driven development teams using Claude Code, this tool helps structure task
decomposition, ensure resumability, and support "Plan Mode" best practices.

---

## 📦 Features

- ✅ Execute structured plans step-by-step
- 🔁 Resume from last checkpoint
- 🧾 View or audit execution logs
- 💾 Store checkpoints per plan run
- ⚙️ Designed for Claude-compatible workflows

---

## 🚀 Quick Start

### 1. Clone or download this repository

```bash
git clone https://github.com/your-org/plan-runner-cli.git
cd plan-runner-cli

2. Install Requirements

No external dependencies required – pure Python 3.7+

⸻

3. Run a New Plan

python3 plan_runner_cli.py --plan "Fetch PRD" "Decompose tasks" "Generate OpenAPI"

📁 A new plan will be created and stored under ./checkpoints/<plan_id>.json

⸻

4. Resume an Existing Plan

python3 plan_runner_cli.py --resume <plan_id>

🔄 You will resume the plan from the last incomplete step.

⸻

📂 Folder Structure

.
├── plan_runner_cli.py       # Main CLI utility
├── README.md                # This file
└── checkpoints/             # Saved plan checkpoints


⸻

🔐 Checkpoint Format

Each checkpoint file (JSON) contains:

{
  "plan_id": "plan_2025_09_01_12_00_00",
  "steps": [
    {"description": "Fetch PRD", "status": "complete"},
    {"description": "Decompose tasks", "status": "pending"},
    ...
  ],
  "last_updated": "2025-09-01T12:05:00"
}


⸻

🛠 Best Practices for Plan Mode (Claude)
	•	Write atomic, verifiable steps
	•	Track each step with a status: "pending", "in_progress", or "complete"
	•	Use checkpoints to support rollback or resume workflows
	•	Embed this utility in larger Claude-based toolchains (e.g., agent runners)

⸻

🧪 Example Use Case

python3 plan_runner_cli.py \
  --plan "Extract entities from PRD" \
         "Generate Clarification Questions" \
         "Validate with GraphRAG" \
         "Produce Draft WBS"


⸻

📜 License

MIT License

⸻

🤝 Contributions

Pull requests, bug reports, and new ideas welcome! Fork and contribute, or open an issue to discuss improvements.

⸻


Would you like me to:
- Add a CLI flag for saving output logs?
- Include example Claude/Agent YAML integration?
```
