# Blind Two-Shopper Ranking

Prompt: `two shoppers talking beside a vegetable stall, in <sks> style`

Holdout seeds: 98100-98104

The model mapping was not read before this ranking was recorded. Criteria were
two-person prompt adherence, facial and full-body coherence, interaction, hands
and feet, vegetable-stall context, and visible artifacts.

| Seed | Ranking | Notes |
| --- | --- | --- |
| 98100 | C > A > B | C gives the foreground shopper the clearest face and pose; all three crop the second shopper. |
| 98101 | A > C > B | A has the most coherent pair and readable faces; B weakens two-person adherence. |
| 98102 | A > B > C | A has the strongest two-person composition and facial balance. |
| 98103 | B > A > C | B has the clearest interaction and most coherent pair. |
| 98104 | B > A > C | B is the only column where both shoppers face each other clearly. |

Anonymous totals: A wins 2, B wins 2, C wins 1.

The mapping can now be decoded.
