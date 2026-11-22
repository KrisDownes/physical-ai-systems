# Previously evaluated rods: now development data

Success = S, failure = F. Method order preserves the original evaluation; historical `history` is the simple-response model in this iteration.

| Case / seed | Offset mm | Fitted prior | Current state | History | Mismatched | Current → history |
|---|---:|:---:|:---:|:---:|:---:|---|
| 0 / 5101 | -85 | F | S | F | S | lost |
| 1 / 5102 | -85 | F | F | S | S | gained |
| 2 / 5103 | -85 | F | F | S | F | gained |
| 3 / 5104 | -55 | S | S | F | F | lost |
| 4 / 5105 | -55 | S | F | S | S | gained |
| 5 / 5106 | -55 | S | F | F | S | unchanged |
| 6 / 5107 | -25 | F | F | S | F | gained |
| 7 / 5108 | -25 | F | F | F | S | unchanged |
| 8 / 5109 | -25 | F | F | F | S | unchanged |
| 9 / 5110 | 25 | F | F | S | F | gained |
| 10 / 5111 | 25 | F | S | F | F | lost |
| 11 / 5112 | 25 | F | F | F | F | unchanged |
| 12 / 5113 | 55 | F | F | S | F | gained |
| 13 / 5114 | 55 | F | S | F | F | lost |
| 14 / 5115 | 55 | F | F | F | F | unchanged |
| 15 / 5116 | 85 | F | F | S | F | gained |
| 16 / 5117 | 85 | F | F | F | F | unchanged |
| 17 / 5118 | 85 | F | F | F | F | unchanged |

For case 1 / seed 5102, the observed starting yaw was 0.2203 rad and ending yaw 0.1237 rad, so the observed change was -0.0967 rad. The starting rod-to-tool lateral separation was 9.1 mm. Current-state selected +49.1 mm and failed; the historical response estimator selected -51.8 mm and succeeded. Evaluator-only true COM was -51.0 mm. The 24 mm-wide support means nominal static support tolerance is about ±12 mm, subject to simulated dynamics. Commands specify displacement; they are not measured force. This is a selected explanatory example, not evidence of generalization.
