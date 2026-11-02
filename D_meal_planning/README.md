# D - Gordon meal planning

Component workspace for joint menu recommendations and planner evaluation. Implementation is planned.

## Deliverables
- Generate combinations of 1–3 Gordon foods at 0.5, 1, or 1.5 listed servings, using complete energy and macronutrient values.
- Apply user preferences, confirmed exclusions and nutrient/energy bounds.
- Rank projected daily totals by normalized energy-target deviation, protein shortfall, and carbohydrate/fat distance from their planning ranges; ties favor fewer foods.
- Return portions, stations, projected totals and remaining gaps; handle infeasible or already-met goals explicitly.
- Compare joint planning against protein-only ranking using identical menu candidates and constraints.

## Interface
Input: component C daily nutrient state, preferences, limits and dated Gordon menu.
Output: ranked meal combinations for user review; recommendations become intake only after consumption confirmation.

Keep component experiments and notes here; reusable logic belongs in ../src/ and reproducible evaluations in ../experiments/. Do not commit personal profiles/logs or private identity mappings.
