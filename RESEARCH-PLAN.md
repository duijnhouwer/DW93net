# DW93NET research plan

Discussion recorded: October 17, 2026.

This document preserves Jacob's research direction and proposed experiments.
It is not an executable experiment specification, a claim that every proposed
comparison has been completed, or authorization to launch training. Numerical
thresholds, mechanism definitions, and compute budgets remain to be decided.

## Scientific objective and career context

Reproducing the Duffy–Wurtz illusion is the guiding scientific objective:
understand when a network trained to estimate self-motion shows the human-like
heading bias elicited by superimposed translation and rotation flow.

Intermediate studies should answer substantive questions on the way to that
objective and may support publications before the illusion is reproduced.
Publication is an aspiration, not guaranteed by completing an experiment.
The goal is not to replace the illusion question with generic estimation accuracy.

Jacob worked intensively on this project before being laid off. He wants to
continue while pursuing employment, using completed computational-vision work,
reproducible experiments, public software, and potentially papers to build a
visible research track record. The job search should not depend on finishing
the entire research program. Python proficiency and personal contributions
should be described accurately, including the role of AI-assisted development.

## Established starting point and evidence to recover

Jacob reports that at least one relatively simple model trained with a 360°
field of view accurately estimates translation and rotation from combined
translation-and-rotation flow. That trained model did not show the desired
human-like illusion when presented with superimposed flow.

This is an existing result, not a prerequisite still to be achieved. Its exact
checkpoint, source revision, training specification, quantitative accuracy, and
superimposed-flow responses need to be recovered and documented before formal
reporting. The October 17 repository review did not independently evaluate the
trained model or inspect its original result artifacts.

The `main` branch inspected initially contained a translation-only recipe and
underrepresented subsequent progress. Relevant later work exists on:

- `first_run_transn_and_rot_recipe`: joint translation and rotation training.
- `Geometry-Aware-Architecture-Learnable-Motion-Processing-Rotational-Weight-Sharing`:
  panoramic processing, mission control, a Duffy–Wurtz evaluation workflow,
  and a proposed architecture redesign. This is the current branch when this
  discussion was recorded.

Run outputs are configured under `/home/jacob/Documents/dw93net_output` on the
Ubuntu system; trained artifacts were not found committed in the inspected tree.
Do not mistake historical throughput benchmarks or pipeline tests for evidence
of the current model's scientific performance.

## Why full-circle vision is a special starting point

Jacob's working explanation is that 360° coverage offers a simple route to
rotation estimation: rotation contributes common signed angular motion around
the circle, whereas translation contributions cancel in the global aggregate.
Reducing coverage removes that cancellation and may require a different
computation, rather than merely more capacity for the same computation.

Exact cancellation depends on scene geometry, depth distribution, sampling, and
motion weighting. The current input is correlation-based rather than an oracle
list of motion vectors. Treat the global-motion shortcut as a hypothesis about
available information and learned behavior, not an established explanation of
the network's internal computation. Test it using the known stimulus geometry
and a simple reference estimator.

The 360° to 270° transition may therefore be more consequential than subsequent
FOV reductions. Performance could nevertheless deteriorate gradually rather
than discontinuously. Additional FOV samples near full coverage were discussed
as an option, not selected as a final sweep. Respect the near-panoramic boundary
limitations documented in `PLAN-PANORAMIQUE.md` when choosing conditions.

## Standard comparison for every reported model

Jacob intends the following comparison to be standard procedure for all models
he publishes. Use the same architecture in both training conditions:

- **YESYAW:** train on combined translation and rotation.
- **NOYAW:** train on translation without rotation.

Evaluate both models on both stimulus classes:

| Training condition | Combined translation and rotation | Superimposed translation and rotation |
| --- | --- | --- |
| YESYAW | Assess learned decomposition | Measure heading bias and illusion-like response |
| NOYAW | Assess response to rotation absent during training | Measure heading bias without rotation training |

Combined flow means one scene transformed by camera translation and rotation.
Superimposed flow means separate translation-only and rotation-only motion
layers. Preserve this distinction in generation, metadata, plots, and reports.

Measure signed heading bias as well as decomposition accuracy; an aggregate
absolute error alone does not characterize an illusion. Define stimulus ranges,
layer proportions, reference headings, and the expected direction of bias
before interpreting responses. NOYAW performance on rotated stimuli is a probe
of generalization, not a trained yaw-recovery capability.

## Proposed first paper: restricted field of view

The first study would explore how translation–rotation decomposition breaks
down as FOV narrows and what is needed to restore it:

1. Preserve and characterize the successful 360° baseline.
2. Seek a simple model that retains reliable 360° decomposition; define the
   simplicity criteria rather than claiming universal minimality.
3. Train matched models at restricted FOVs, provisionally 270°, 180°, and 120°.
4. Characterize failures and test predefined, biologically motivated mechanisms
   that might restore decomposition at smaller FOVs.
5. Apply the standard YESYAW/NOYAW and combined/superimposed comparison throughout.

The scope is a characterization of computational requirements within a defined
model family, together with the models' illusion responses. It can yield a
substantive contribution even if the human-like illusion has not yet emerged.

Control or explicitly study visible dot count versus dot density, scene depth,
angular sampling, motion ranges, noise, training budget, and boundary behavior.
A reference estimator supplied with true local motion could help distinguish
motion-extraction limitations from decomposition or stimulus-information limits.

## Curriculum learning as a planned comparison

Jacob proposes starting with a successful 360° model, continuing training from
its weights at 270°, then continuing at 180°, potentially followed by 120°.
Save and evaluate each stage under the standard protocol.

Compare this with training from scratch at the target FOV. Account for total
training exposure and compute so that extra training is not confused with a
curriculum benefit. Evaluate immediately after narrowing the FOV and again
after adaptation to distinguish transfer from additional learning.

The full-circle solution may supply useful representations or encourage a
shortcut that must be unlearned. Both outcomes are informative.

Weight-transfer semantics need an explicit design. Keeping tensor dimensions
compatible is insufficient if changing FOV changes the angles represented by
spatial bins. The current adaptive pooling can retain output dimensions while
changing those meanings. Fixed angular layouts with explicit visibility masks
were discussed as one possible approach, not an approved implementation.

## Mechanism comparisons: factorial rather than only sequential

Base architectural additions on known properties of the visual system, beginning
with a small subset. Define the biological motivation, computational operation,
and predicted benefit of each mechanism before running the comparisons.

Jacob clarified that the intended family includes combinations such as:

- A
- A+B
- A+C
- A+B+C

A is a common baseline; B and C are optional mechanisms. A single nested ladder
cannot distinguish independent benefits, substitution, and interactions.
Factorial comparisons can, subject to valid dependencies between mechanisms.
With k optional mechanisms there are 2^k combinations before multiplying by FOV,
training condition, seeds, and training history.

Use precise interpretations within the tested family and conditions:

- **Sufficient:** a mechanism set reliably meets the predefined criterion.
- **Necessary within a model family/condition:** matched removal and retraining
  prevent the criterion being met under the specified training budget. Failed
  optimization is not proof of mathematical impossibility.
- **Redundant or interchangeable:** alternative mechanism sets perform similarly.
- **No detected benefit:** a mechanism has no meaningful measured benefit in the
  tested conditions; do not call it universally irrelevant.

Removal without retraining probes dependence of a particular trained solution;
removal with retraining probes what the reduced architecture can learn. Neither
alone establishes necessity in the biological visual system.

## Containing hyperparameter and decoder complexity

The decoder ends in three continuous outputs: Dx, Dz, and Rot. Its number of
fully connected hidden layers and units per layer introduces another potentially
large search space. A powerful decoder may perform computations otherwise
attributed to the proposed visual mechanisms.

The following approach was suggested, but exact settings are undecided:

1. Separate scientific variables (mechanisms, FOV, training history) from tuning
   choices (widths, learning rates, and optimization settings).
2. Use a small decoder-capacity ladder: linear readout, one hidden layer, then
   two hidden layers, initially with a fixed selected width.
3. Compare mechanisms at matched decoder settings and with equal predefined
   tuning budgets. Include capacity controls where added parameter count could
   explain a mechanism's apparent benefit.
4. Apply broader width/depth sensitivity checks to decisive comparisons instead
   of exhaustively tuning every architecture.

A mechanism that helps a linear readout but not a nonlinear decoder may make
information more accessible rather than uniquely enable decomposition. Treat
decoder capacity as part of the interpretation, not merely an implementation detail.

No layer widths, mechanism list, training seeds, success thresholds, or exhaustive
hyperparameter grid have been selected in this discussion.

## Unattended experiment orchestration

Jacob wants AI-assisted work to run overnight with minimal interaction so he
can pursue other activities. The existing system runs one paradigm at a time.
A proposed orchestration layer would systematically queue experiments above
that runner; sequential execution is sufficient initially.

Desired capabilities:

- Persistent experiment specifications, queue state, dependencies, and budgets.
- Exact source, configuration, dataset, seed, checkpoint, and parent-checkpoint
  identity for each run, including curriculum ancestry.
- Completion checks, resumption, and a distinction between technical failure
  and scientifically poor performance.
- Standard evaluations and a morning report of results, failures, and pending work.
- Predetermined rules for repeats, permitted follow-ups, and stopping.
- Preservation of prior artifacts and no source mutation during active runs.

Begin with an agreed fixed comparison batch. Adaptive architecture selection
can follow once its allowed choices and decision rules are explicit. This
document does not authorize unconstrained autonomous architecture search.

Select decomposition models using validation data. Report illusion responses
separately; if those responses guide subsequent model design or selection,
recognize that as a fitting stage and reserve fresh confirmatory conditions.

## Next planning decisions

1. Recover the successful 360° run and its original illusion evaluation.
2. Specify the baseline, candidate mechanisms, and valid combinations.
3. Define quantitative decomposition and illusion metrics, success criteria,
   repeat seeds, and uncertainty reporting.
4. Choose the initial FOV sweep, decoder settings, and bounded tuning policy.
5. Decide where curriculum comparisons enter and define weight-transfer semantics.
6. Turn the agreed first batch into executable experiment specifications and
   identify the smallest required mission-control extension.

## Related documents

- `README.md`: scientific background and current operational workflow.
- `Geometry-Aware-Architecture-Learnable-Motion-Processing-Rotational-Weight-Sharing-PLAN.md`:
  a specific proposed architecture; candidate ideas rather than an implemented
  factorial research program.
- `PLAN-PANORAMIQUE.md`: panoramic geometry and boundary implementation plan.
- `HANDOFF.md`: historical panoramic implementation handoff; not a current
  record of every subsequent scientific experiment.

This broader research plan does not silently supersede those technical documents
or imply that their proposed mechanisms have already been implemented.
