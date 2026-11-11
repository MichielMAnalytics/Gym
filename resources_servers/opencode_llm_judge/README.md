# OpenCode tool rewards with the generic LLM judge

This training component wraps `equivalence_llm_judge` and retains its task-data
schema, answer extraction, rescue and swap checks. Tool rewards use the existing
OpenCode execution evidence. Correctness and tool rewards are recorded separately;
malformed, conflicting or incomplete judge replies are masked.

The shared defaults apply only penalties, only on correct answers: failed bash
calls cost 0.05, consecutive duplicate calls cost 0.02, and the third through sixth
bash attempts incur total usage penalties of 0.025, 0.05, 0.075 and 0.10. The usage
penalty stays capped at 0.10. Incorrect answers receive no tool adjustment.

Compose `configs/opencode_llm_judge.yaml` with a separate `training_judge` model
and an environment targeting `llm_judge_with_tools`. Enable
`forward_agent_evidence: true` on that environment and separately on reverification
when recomputing tool rewards. Default judge prompt paths resolve independently
of the component working directory.

The wrapper imports the shared grader and benchmark reward helper; neither the
generic grader nor the visual task schema depends on this component. Dataset rows
use the generic judge's `expected_answer` contract, with no visual metadata required.
See the [harness integration guide](../../fern/versions/latest/pages/agent-server/integrate-existing-agents.mdx).
