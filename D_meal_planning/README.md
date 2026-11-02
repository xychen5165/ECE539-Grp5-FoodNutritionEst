# D - Gordon meal planning

Component workspace for joint menu recommendations and planner evaluation. Implementation is planned.

## Deliverables
- Generate bounded food/portion combinations from Gordon menu entries and their supplied nutrition.
- Apply user preferences, confirmed exclusions and nutrient/energy bounds.
- Rank combinations by remaining protein and, only with complete intake coverage, fiber goals.
- Return portions, stations, projected totals and remaining gaps; handle infeasible or already-met goals explicitly.
- Compare joint planning against protein-only ranking using identical menu candidates and constraints.

## Interface
Input: component C daily nutrient state, preferences, limits and dated Gordon menu.
Output: ranked meal combinations for user review; recommendations become intake only after consumption confirmation.

Keep component experiments and notes here; reusable logic belongs in ../src/ and reproducible evaluations in ../experiments/. Do not commit personal profiles/logs or private identity mappings.
