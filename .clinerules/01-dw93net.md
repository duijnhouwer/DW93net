# Current task: stimulus generation

- Focus only on getting the stimulus generator running and creating
  the stimuli Jacob requests. The inverse estimator is out of scope.
- Read only the files needed for this task. Do not repeat whole-repo
  discovery, review all Git history, or propose architectural changes.
- Implement clear requests directly. Ask only when a missing decision
  affects stimulus correctness.
- Preserve unrelated work. Do not commit, push, or delete files unless asked.
- Preserve intended stimulus values, units, coordinate conventions,
  and output labels. Verify numerical scaling when changing units.
- Run a small generation check before producing the requested batch.
  Report what was generated and where it was saved, then stop.
- Keep responses concise. Qwen is text-only; do not send it images.

# Execution environment

- Use /home/jacob/venvs/nvidia/bin/python.
- Reserve the AMD R9700 for Qwen. Do not use it for stimulus generation
  or change Python environments or GPU visibility settings.
- If GPU computation is needed, verify CUDA is available,
  torch.version.cuda is set, torch.version.hip is None, and the
  selected device is the RTX 3080 Ti. Stop if these checks fail.