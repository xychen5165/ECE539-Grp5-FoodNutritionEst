# References

These references are transcribed from the November 1, 2026 team proposal, *Photo-Based Daily Nutrition and Gordon Meal Planning*. Numbers match the citations in the [architecture plan](architecture.md). They support the component choices and evaluation methods.

1. **UW Housing Dining. Gordon Nutrislice menu.** Serving and nutrition data for the menu client and planner. Accessed October 29, 2026 in the proposal. [Menu](https://wisc-housingdining.nutrislice.com/menu/gordon-avenue-market).
2. **Kirillov et al. Segment Anything. ICCV, 2023.** Frozen SAM segmentation in component A. [Paper](https://arxiv.org/abs/2304.02643).
3. **Radford et al. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021.** Frozen CLIP features in component B. [Paper](https://arxiv.org/abs/2103.00020).
4. **Thames et al. Nutrition5k. CVPR, 2021.** RGB nutrition regression data and official partitions. [Paper](https://arxiv.org/abs/2103.03375), [dataset and splits](https://github.com/google-research-datasets/Nutrition5k).
5. **Health Canada. DRI macronutrient reference values and 2023 EER equations.** Initial daily energy and macronutrient targets in component C. [Macronutrients](https://www.canada.ca/en/health-canada/services/food-nutrition/healthy-eating/dietary-reference-intakes/tables/reference-values-macronutrients.html), [energy equations](https://www.canada.ca/en/health-canada/services/food-nutrition/healthy-eating/dietary-reference-intakes/tables/equations-estimate-energy-requirement.html).
6. **NIH ODS. Exercise and Athletic Performance.** Reference for user-confirmed exercise-related protein adjustments. [Fact sheet](https://ods.od.nih.gov/factsheets/ExerciseAndAthleticPerformance-HealthProfessional/).
7. **NIDDK. Body Weight Planner.** Reference for user-confirmed goal-specific energy adjustments. [Resource](https://www.niddk.nih.gov/health-information/weight-management/body-weight-planner).
8. **Zhao et al. Visual nutrition analysis: leveraging segmentation and regression for food nutrient estimation. Frontiers in Nutrition, 2024.** Proposed food-region annotations for segmentation evaluation; matching development images remains pending. [Paper](https://doi.org/10.3389/fnut.2024.1469878).

## Earlier brief references

The brief also listed [Maharsh Suryawala's Food Image Recognition](https://github.com/MaharshSuryawala/Food-Image-Recognition) and [Hand et al. (2026)](https://doi.org/10.3389/fnut.2026.1927530). They remain background research leads from the brief; verify their details and relevance before citing them in later reports.
