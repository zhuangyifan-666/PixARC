# SpeCa paper versus released code

The registered baseline follows the local released SpeCa-DiT code at Cache4Diffusion commit `91a1949fcc88acab46547f0b5f295f5de2df2870`. The paper is useful for motivation, but prose is not silently substituted for executable behavior. No paper PDF is vendored in this checkout; the paper-described column records the version linked by the local [`Cache4Diffusion README`](../../../baselines/Cache4Diffusion/README.md), while local source remains the executable authority.

## Three distinct contracts

| Topic | Paper-described behavior | Released-code behavior | JiT adaptation behavior |
|---|---|---|---|
| draft | TaylorSeer predicts multiple future model states | one online Taylor decision per current NFE, with per-block attention/MLP factors | same released online semantics, two CFG histories |
| error | global relative L2 norm ratio, paper epsilon `1e-8` | defaults to elementwise batch-global `relative_l1`, epsilon `1e-10` | same released metric over mathematically concatenated cond/uncond payloads |
| validation | sequentially accept future predictions while error is within threshold | one local exact check at the current Taylor NFE | one local check at the shared current JiT NFE |
| rejected prediction | prose describes rejection/reversion of the offending future prediction | current speculative output is kept; no rollback or replay | current cond/uncond draft outputs feed CFG/Heun unchanged |
| exact branch input | naturally read as an exact target | exact block starts from the current speculative prefix | exact final-block branch starts from the same stream-specific speculative input |
| exact writeback | reject semantics suggest exact recovery | verifier output is metric-only | metric-only, never a Taylor anchor |
| scheduling error | current candidate acceptance | newly computed error is stored and consulted on the next NFE | `begin_nfe` reads previous error; `end_nfe` writes current error |
| final layer | final Transformer layer/block | hard-coded block 27 in a 28-block DiT | `verify_layer=-1` resolves `len(blocks)-1` (block 11 for JiT-B/16) |
| threshold | `base * decay**progress` | same form plus floor `0.01`; first progress is `1/N` | same formula/floor on monotonic Heun NFE coordinate |
| interval | not the adaptive decision rule | parsed/mentioned inconsistently and unused by `cal_type` | `null`/ignored in `speca`; active only in fixed-draft ablation |

## Metric formulas

Paper-style global relative L2 is conceptually a norm ratio such as `||pred-exact||_2 / (||exact||_2 + eps)`. Released `relative_l2` instead computes the RMS of elementwise relative errors. Released main `relative_l1` is:

```text
mean(abs(pred - exact) / (abs(exact) + 1e-10))
```

For the toy vectors `pred=[3,1]`, `exact=[1,2]`, the released functions produce L1 `1.5`, L2 `1.581138849`, relative L1 `1.25`, relative L2 `1.457737923`, and cosine error `0.292893291`. A global-norm relative L2 would be `1.0`, demonstrating that the definitions are not interchangeable.

## Consequences for claims

- A `verification_fail_at_current_nfe` is not a rejected current solver step. The appropriate later event is `next_nfe_forced_full_due_previous_failure`.
- Verification measures local block discrepancy conditional on a speculative prefix, not a full exact-model counterfactual error.
- The current exact verifier output does not repair the current image trajectory and must not populate history.
- JiT's `official_nfe_index` is an adaptation required by Heun's repeated continuous times, not paper semantics.
- Quality and speed claims must describe this released-code-faithful system. They cannot imply paper-style rollback, per-sample ragged scheduling, or an exact global verifier.

`paper_semantics_experimental` is reserved for a future, explicitly authorized ablation. It is not implemented as the main runtime, not used by the provided configs, and would require a separate correctness/fairness study.
