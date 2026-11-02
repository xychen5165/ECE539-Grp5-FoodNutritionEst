# C - Daily nutrition tracking

Component workspace for reference targets, confirmed intake records and menu access. Implementation is planned.

## Deliverables
- Implement sourced, editable reference targets from profile and goal inputs.
- Apply user-confirmed consumption fractions, record meals/snacks and calculate daily totals and remaining goals.
- Support corrections without double-counting; identify incomplete logs and unknown nutrients.
- Fetch Gordon menus by America/Chicago date and selected breakfast/lunch/dinner; cache dated responses and validate nutrient units.
- Test calculation rules, date rollover, meal switching and record updates.

## Interface
Input: component B nutrient estimates, confirmed consumption, manual records and user profile/goals.
Output: daily totals, remaining targets/bounds and validated Gordon menu data for component D.

Generic meal-photo estimates are independent of Gordon menu recipes. Keep component notes here and reusable code in ../src/. Profiles, food logs and private identity mappings must stay outside this public repository.
