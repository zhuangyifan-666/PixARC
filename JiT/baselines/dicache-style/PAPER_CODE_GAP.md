# Paper versus released code

The paper describes the method; the released scripts add backend-specific scheduling needed for reproduction.

| Topic | Paper / Algorithm 1 | Released code used here |
|---|---|---|
| Primary probe error | `delta_y` in Eq. 5 | FLUX supports `delta_y` and `delta_minus`; demo selects `delta_y` |
| Gate equality | pseudocode uses `<=` reuse | FLUX and WAN use strict `<`; Hunyuan uses `<=` |
| Initial Full policy | first evaluation is Full | FLUX also has inclusive `ret_ratio` warmup and final Full |
| DCTA history | two exact anchors in Eqs. 7–11 | released code appends histories and reads their latest two |
| DCTA order | first-order main method | higher order appears as an ablation; JiT main stays first order |
| Numerical stabilization | no epsilon prescription | released formulas divide directly; stable epsilon is a named ablation only |
| Gamma safety | formula/clip | backend-specific ranges; non-finite policy is unspecified |
| CFG/batch/reset | not a full systems contract | released backends differ substantially |

The executable main profile therefore follows released FLUX where the paper is underspecified: strict `<`, inclusive warmup, final Full, adjacent-call observations, exact-only anchors, first-order DCTA, and gamma `[1,1.5]` without epsilon.

## Required clarifications

1. The paper’s main probe error is `delta_y`.
2. For current/previous tensors with identical shape and element count, `mean(abs(diff))/mean(abs(reference))` equals the corresponding L1-norm ratio because the common `1/numel` factors cancel.
3. `delta_minus` is a released-code option, not the paper’s main formula.
4. The paper does not prescribe released FLUX’s `ret_ratio` warmup.
5. It does not prescribe forced last Full.
6. It does not define the released backend gamma clipping policy as a universal rule.
7. Consequently it supplies no universal gamma upper bound.
8. FLUX’s 1.5 versus WAN’s 2.0 is a backend/release choice; it cannot be derived as one paper constant.
9. No epsilon or non-finite recovery policy is specified.
10. Independent CFG branch decisions are not specified.
11. Batch-global versus per-sample gate reduction is not specified.
12. “Sample-specific” describes trajectory intent; released tensor-wide means become genuinely sample-specific at B=1, while B>1 is grouped-batch behavior.
13. First-order equations use two exact anchors, but the paper does not mandate a particular deque implementation.
14. FLUX lists grow without bound although only the latest two are used; WAN/Hunyuan can retain short stale state. The bounded port removes behaviorally unused history.
15. The paper motivates a cheap probe and reports method-level efficiency, but does not provide this JiT port’s probe/gate/scalar/DCTA runtime breakdown.
16. Monkey patching and class-level state are released engineering choices, not algorithm requirements.

The paper and release do not define a JiT threshold, a CUDA compilation strategy, a non-finite gamma recovery policy, deterministic 50K sharding, or cross-backend batch semantics. This port exposes those as explicit configuration/metadata. Both `rel_l1_thresh` and `gamma_nonfinite_policy` remain null in the candidate template until preregistered on disjoint validation data; the generator and deferred guard reject unresolved values.
