# Experiments

Keep each experiment's settings, optional notebook, and small results together. Use names such as `001-baseline` and `002-<technical-change>` once the experiment is defined.

Copy [run-template.md](run-template.md) into the experiment directory as a run record. For repeated runs, use distinct run IDs and separate result paths. Update this index with links, status, and a short verified conclusion.

Shared implementation belongs in `src/`. Notebooks and configuration files are optional; exact reproducible settings and commands are required for reported results. Store large artifacts externally and document access.

The evaluation plan includes a training-set median baseline, an original-versus-SAM-masked image comparison, and joint meal ranking versus protein-only ranking. See the [evaluation plan](../docs/architecture.md#training-and-evaluation). Add run records and measured results here as experiments are executed.
