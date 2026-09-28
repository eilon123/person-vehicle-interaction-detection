# Annotation decisions and review notes

These notes precede automated prediction review. Contact sheets are navigation
aids; proposed actions/bounds need dense temporal review before becoming reference
annotations. They are not accuracy ground truth.

| Clip | Observation / ambiguity | Decision and alternative |
|---|---|---|
| 1THkHYIQ_bY_0 | A person removes a large cover from a parked car. | Directed handling of a vehicle cover counts as `other_interaction`; exclude later movement away carrying the cover. Alternative: restrict to cabin actions, which would miss a clear vehicle-directed interaction. |
| gt1125_06 | Aerial view, moving traffic and very small people. | Use full-resolution crops for review; no interaction cannot be concluded from a downscaled contact sheet. |
| HIu4lM4B8hA_1 | A person is partly occluded beside an already-open door at the start; later moves behind the car. | Do not infer entry solely from disappearance. Separate observable door/contact action from uncertain cabin transition. |
| iMGR_0AG3a8_2_3 | Multiple people and cars; vehicles occlude people. | Preserve competing pair candidates. Pair proximity alone is insufficient. |
| mKzCQKTHizw_0 | A person approaches an open passenger door near clip end; camera pans. | Mark a visible partial entry as truncated, if supported by dense review. Do not extend its interval into another clip. |
| mKzCQKTHizw_1 | Person approaches, enters silver car, and door closes. | Include door closure in entry; do not separately count it as door operation. Keep the two same-prefix clips in one evaluation source group. |
| NmlzoaDcOuI_1 | White-shirted person emerges beside a silver car; other pedestrians pass. | Distinguish exit from passersby; later far-side action requires separate review. |
| NmlzoaDcOuI_6 | Green-shirted person is outside a red car with door open at clip start, leans into cabin, then closes door. | Outside leaning is not full entry. Report supported contact/door actions rather than infer an unseen preceding exit. |

Clothing/color labels describe appearance only. Monochrome/night footage does not
support confident physical color claims. Local IDs reset per clip and scene cut.
