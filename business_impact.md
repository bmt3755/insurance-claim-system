================================================================================
# BUSINESS IMPACT LOG — Insurance Claim Processing Agent
================================================================================

Every architecture decision is documented here.
Format: Technical reason -> Business consequence -> Risk/cost impact (category and direction).
No fabricated dollar figures. No measured metrics are claimed unless the code produces them.

--------------------------------------------------------------------------------

## DECISIONS

--------------------------------------------------------------------------------
### Decision 1: Typed ClaimState with strict per-section ownership
Technical reason: ClaimState is a TypedDict divided into four sections — input, routing, processing, tracking. The router writes only to routing. Each specialist writes only to processing. No agent reads or writes another agent's section. Section types are enforced at definition time in schema.py.
Business consequence: Without section boundaries, any agent can overwrite any other agent's output. In a pipeline where two agents run in sequence, the second agent can silently corrupt the first agent's result — producing a wrong decision with no trace of what happened. Typed sections make every field's owner explicit.
Risk/cost impact: AUDIT TRACEABILITY — improves. Each decision is attributable to exactly one agent's input and output. In a regulated environment, decision traceability is not optional.

--------------------------------------------------------------------------------
### Decision 2: Pydantic validation of all LLM outputs before state write
Technical reason: Each agent defines a private Pydantic BaseModel (_RouterScores, _AutoClaimScores, etc.) with Field(..., ge=0, le=10) on every score. The raw LLM JSON response is parsed through model_validate_json() before any value is read. An out-of-range score raises a ValidationError and lands in the except block.
Business consequence: Without validation, an LLM returning a score of 11 or "N/A" silently writes a bad value into state. Downstream decision logic (score > 7.0 maps to APPROVE) then operates on garbage input. The decision looks valid but was produced from invalid data.
Risk/cost impact: DATA INTEGRITY — improves. Prevents corrupted LLM outputs from reaching decision logic silently.

--------------------------------------------------------------------------------
### Decision 3: Confidence threshold AND ambiguity margin gate in the router
Technical reason: _pick_claim_type() in router.py enforces two independent conditions before routing: top_score >= 6.0 (minimum confidence) AND (top_score - second_score) > 2.0 (not ambiguous). If either fails, the function returns None and the router writes claim_type=None to state. The claim does not reach any specialist.
Business consequence: A claim scoring auto=7.1 and health=7.0 would be force-routed to the auto specialist by a single-threshold gate — a 0.1-point margin that is not a confident classification. The auto specialist then runs vehicle damage checks on a claim that may be medical. The resulting score and decision are wrong before any check is performed.
Risk/cost impact: CLASSIFICATION ERROR — reduces. Borderline claims route to human review rather than to the wrong specialist.

--------------------------------------------------------------------------------
### Decision 4: Three-tier decision with NEEDS_REVIEW band (5.0-7.0)
Technical reason: _map_score_to_decision() in each specialist maps overall_score to three outcomes: above 7.0 = APPROVE, below 5.0 = DENY, 5.0 to 7.0 inclusive = NEEDS_REVIEW. The function comment explicitly notes "boundary values go to NEEDS_REVIEW (safer for compliance)."
Business consequence: A binary approve/deny gate forces a decision on every claim, including ones where the evidence is genuinely mixed. The NEEDS_REVIEW band is where legitimate edge cases live — partial documentation, unusual circumstances, scores in the middle because they are unclear, not because they are fraudulent. Without this band, the system is making autonomous adverse decisions on ambiguous inputs.
Risk/cost impact: WRONGFUL AUTOMATED DECISION — reduces. Human review is preserved exactly where the model is uncertain.

--------------------------------------------------------------------------------
### Decision 5: NEEDS_REVIEW as the error fallback — never DENY
Technical reason: Every specialist agent wraps its LLM call in try/except Exception. On any exception — network failure, parse error, unexpected response format — the except block returns decision=Decision.NEEDS_REVIEW.value, agent_status=FAILED, and increments errors_encountered in tracking. The claim is never autonomously denied due to a software failure.
Business consequence: A bug-triggered DENY is indistinguishable from a legitimate denial in the output. It is produced by a failure, not by evaluating the claim. If this happens at scale — a network timeout, a model outage, a bad deployment — a batch of claims is wrongfully denied with no indication in the decision field.
Risk/cost impact: WRONGFUL AUTOMATED DENIAL — eliminates the failure mode. Failed claims always route to human review.

--------------------------------------------------------------------------------
### Decision 6: temperature=0 on all LLM calls
Technical reason: Every agent passes temperature=0 to the OpenAI completions API. This instructs the model to produce the highest-probability token at each step, making the output deterministic for a given input. The same claim text submitted twice will produce the same scores and the same decision.
Business consequence: A non-deterministic system can route the same claim differently on two submissions — to different specialists, producing different checks and different decisions. Two different decisions on identical input cannot be explained or defended. It also breaks any attempt at regression testing with fixed inputs.
Risk/cost impact: DECISION INCONSISTENCY — eliminates. Same input always produces same output.

--------------------------------------------------------------------------------
### Decision 7: LangSmith wrap_openai on all LLM calls
Technical reason: Every agent initializes _client = wrap_openai(OpenAI()). This wraps the OpenAI client so every API call is automatically captured as a span in LangSmith — the exact prompt sent, the exact response received, the model used, and the latency. No additional instrumentation is required per call.
Business consequence: Without tracing, there is no record of what the model was given or what it returned for a specific claim. When a decision is disputed — in a customer complaint, an internal audit, or a regulatory review — the only available evidence is the final decision value. The reasoning that produced it is gone.
Risk/cost impact: DECISION AUDITABILITY — improves. Every LLM call is reconstructable from the LangSmith trace.

--------------------------------------------------------------------------------
### Decision 8: LangGraph-compatible node signatures on all agents
Technical reason: Every agent function signature follows the LangGraph node convention: accepts ClaimState as input, returns a dict of updated state sections. This is enforced by the type annotations (state: ClaimState) -> dict. No graph is built in this repository, but the functions are drop-in compatible with StateGraph.add_node().
Business consequence: Agent functions designed as standalone callables with ad-hoc input/output formats cannot be plugged into a graph framework without rewriting them. By adopting the node convention from the start, the codebase can be extended to a full LangGraph pipeline — with checkpointing, conditional edges, and interrupt support — without touching the agent logic.
Risk/cost impact: FUTURE EXTENSIBILITY — preserves. Adding a graph runner, parallel execution, or human-in-the-loop interrupts is additive work, not a rewrite.

--------------------------------------------------------------------------------

## SUMMARY TABLE

| Decision | Risk Category | Direction |
|---|---|---|
| Typed section ownership | Audit traceability | Improves |
| Pydantic LLM output validation | Data integrity | Improves |
| Confidence + ambiguity gate | Classification error | Reduces |
| NEEDS_REVIEW score band (5.0-7.0) | Wrongful automated decision | Reduces |
| NEEDS_REVIEW error fallback | Wrongful automated denial | Eliminates |
| temperature=0 | Decision inconsistency | Eliminates |
| LangSmith wrap_openai tracing | Decision auditability | Improves |
| LangGraph-compatible node signatures | Future extensibility | Preserves |

--------------------------------------------------------------------------------
