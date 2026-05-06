# Eval Sample Audit

Heuristic audit for empty responses, invalid filtered responses, long outputs, and cases where the target appears in the raw response but exact_match is zero.

## Summary

| Variant | Task | File | N | Exact | Empty | Invalid | Target Seen | Extract Miss? | Long | Avg Chars |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma-4-E4B-gguf-q8 | gsm8k_cot_zeroshot | samples_gsm8k_cot_zeroshot_2026-05-06T01-45-34.806046.jsonl | 100 | 0.35 | 0 | 50 | 78 | 44 | 0 | 665.5 |
| gemma-4-E4B-gguf-q8 | hrm8k | samples_hrm8k_gsm8k_2026-05-06T02-45-26.939493.jsonl | 50 | 0.74 | 2 | 0 | 39 | 2 | 0 | 463.4 |
| gemma-4-E4B-gguf-q8 | hrm8k | samples_hrm8k_ksm_2026-05-06T02-45-26.939493.jsonl | 50 | 0.02 | 39 | 0 | 6 | 6 | 0 | 72.2 |
| gemma-4-E4B-gguf-q8 | hrm8k | samples_hrm8k_math_2026-05-06T02-45-26.939493.jsonl | 50 | 0.52 | 7 | 0 | 32 | 6 | 0 | 415.3 |
| gemma-4-E4B-gguf-q8 | hrm8k | samples_hrm8k_mmmlu_2026-05-06T02-45-26.939493.jsonl | 50 | 0.28 | 19 | 0 | 29 | 15 | 0 | 324.3 |
| gemma-4-E4B-gguf-q8 | hrm8k | samples_hrm8k_omni_math_2026-05-06T02-45-26.939493.jsonl | 50 | 0.0 | 46 | 0 | 1 | 1 | 0 | 22.9 |
| gemma-4-E4B-gguf-q8 | leaderboard_ifeval | samples_leaderboard_ifeval_2026-05-06T02-54-36.410078.jsonl | 50 |  | 0 | 0 | 10 | 0 | 10 | 1688.7 |
| qwen-3.5-4b-gguf-q8 | gsm8k_cot_zeroshot | samples_gsm8k_cot_zeroshot_2026-05-06T03-06-01.268924.jsonl | 100 | 0.12 | 54 | 77 | 28 | 16 | 0 | 213.5 |
| qwen-3.5-4b-gguf-q8 | hrm8k | samples_hrm8k_gsm8k_2026-05-06T04-03-13.691415.jsonl | 50 | 0.2 | 33 | 0 | 12 | 2 | 0 | 75.8 |
| qwen-3.5-4b-gguf-q8 | hrm8k | samples_hrm8k_ksm_2026-05-06T04-03-13.691415.jsonl | 50 | 0.02 | 50 | 0 | 0 | 0 | 0 | 0 |
| qwen-3.5-4b-gguf-q8 | hrm8k | samples_hrm8k_math_2026-05-06T04-03-13.691415.jsonl | 50 | 0.14 | 30 | 0 | 16 | 9 | 0 | 106.9 |
| qwen-3.5-4b-gguf-q8 | hrm8k | samples_hrm8k_mmmlu_2026-05-06T04-03-13.691415.jsonl | 50 | 0.08 | 36 | 0 | 9 | 5 | 0 | 57.3 |
| qwen-3.5-4b-gguf-q8 | hrm8k | samples_hrm8k_omni_math_2026-05-06T04-03-13.691415.jsonl | 50 | 0.0 | 50 | 0 | 0 | 0 | 0 | 0 |
| qwen-3.5-4b-gguf-q8 | leaderboard_ifeval | samples_leaderboard_ifeval_2026-05-06T04-42-44.524614.jsonl | 50 |  | 32 | 0 | 4 | 0 | 1 | 354.5 |
| qwen-3.5-9b-gguf-q8 | gsm8k_cot_zeroshot | samples_gsm8k_cot_zeroshot_2026-05-04T22-18-39.693095.jsonl | 40 | 0.075 | 22 | 32 | 8 | 5 | 0 | 201.3 |
| qwen-3.5-9b-gguf-q8 | hrm8k | samples_hrm8k_gsm8k_2026-05-04T22-56-01.959646.jsonl | 20 | 0.15 | 14 | 0 | 4 | 1 | 0 | 59.1 |
| qwen-3.5-9b-gguf-q8 | hrm8k | samples_hrm8k_ksm_2026-05-04T22-56-01.959646.jsonl | 20 | 0.0 | 20 | 0 | 0 | 0 | 0 | 0 |
| qwen-3.5-9b-gguf-q8 | hrm8k | samples_hrm8k_math_2026-05-04T22-56-01.959646.jsonl | 20 | 0.2 | 13 | 0 | 6 | 2 | 0 | 76.0 |
| qwen-3.5-9b-gguf-q8 | hrm8k | samples_hrm8k_mmmlu_2026-05-04T22-56-01.959646.jsonl | 20 | 0.0 | 19 | 0 | 1 | 1 | 0 | 9.8 |
| qwen-3.5-9b-gguf-q8 | hrm8k | samples_hrm8k_omni_math_2026-05-04T22-56-01.959646.jsonl | 20 | 0.0 | 20 | 0 | 0 | 0 | 0 | 0 |
| qwen-3.5-9b-gguf-q8 | leaderboard_ifeval | samples_leaderboard_ifeval_2026-05-04T23-18-16.608999.jsonl | 20 |  | 10 | 0 | 1 | 0 | 0 | 407.1 |
| qwen-3.6-35b-a3b-gguf-q4 | gsm8k_cot_zeroshot | samples_gsm8k_cot_zeroshot_2026-05-04T23-22-56.992045.jsonl | 40 | 0.025 | 32 | 37 | 2 | 1 | 0 | 63.4 |
| qwen-3.6-35b-a3b-gguf-q4 | hrm8k | samples_hrm8k_gsm8k_2026-05-04T23-43-44.020172.jsonl | 20 | 0.1 | 17 | 0 | 2 | 0 | 0 | 34.5 |
| qwen-3.6-35b-a3b-gguf-q4 | hrm8k | samples_hrm8k_ksm_2026-05-04T23-43-44.020172.jsonl | 20 | 0.0 | 20 | 0 | 0 | 0 | 0 | 0 |
| qwen-3.6-35b-a3b-gguf-q4 | hrm8k | samples_hrm8k_math_2026-05-04T23-43-44.020172.jsonl | 20 | 0.05 | 16 | 0 | 3 | 2 | 0 | 36.9 |
| qwen-3.6-35b-a3b-gguf-q4 | hrm8k | samples_hrm8k_mmmlu_2026-05-04T23-43-44.020172.jsonl | 20 | 0.05 | 19 | 0 | 1 | 0 | 0 | 7.6 |
| qwen-3.6-35b-a3b-gguf-q4 | hrm8k | samples_hrm8k_omni_math_2026-05-04T23-43-44.020172.jsonl | 20 | 0.0 | 20 | 0 | 0 | 0 | 0 | 0 |
| qwen-3.6-35b-a3b-gguf-q4 | leaderboard_ifeval | samples_leaderboard_ifeval_2026-05-04T23-54-35.256254.jsonl | 20 |  | 4 | 0 | 4 | 0 | 2 | 990.2 |
| qwen-3-coder-next-gguf-q4 | gsm8k_cot_zeroshot | samples_gsm8k_cot_zeroshot_2026-05-06T01-00-54.829126.jsonl | 100 | 0.43 | 0 | 50 | 92 | 50 | 0 | 770.5 |
| qwen-3-coder-next-gguf-q4 | hrm8k | samples_hrm8k_gsm8k_2026-05-06T05-15-08.385303.jsonl | 50 | 0.9 | 0 | 0 | 49 | 4 | 0 | 426.3 |
| qwen-3-coder-next-gguf-q4 | hrm8k | samples_hrm8k_ksm_2026-05-06T05-15-08.385303.jsonl | 50 | 0.02 | 0 | 0 | 32 | 31 | 0 | 908.8 |
| qwen-3-coder-next-gguf-q4 | hrm8k | samples_hrm8k_math_2026-05-06T05-15-08.385303.jsonl | 50 | 0.74 | 0 | 0 | 45 | 8 | 0 | 544 |
| qwen-3-coder-next-gguf-q4 | hrm8k | samples_hrm8k_mmmlu_2026-05-06T05-15-08.385303.jsonl | 50 | 0.46 | 0 | 0 | 45 | 22 | 0 | 695.4 |
| qwen-3-coder-next-gguf-q4 | hrm8k | samples_hrm8k_omni_math_2026-05-06T05-15-08.385303.jsonl | 50 | 0.02 | 0 | 0 | 16 | 15 | 0 | 837.9 |
| qwen-3-coder-next-gguf-q4 | leaderboard_ifeval | samples_leaderboard_ifeval_2026-05-06T04-50-13.490010.jsonl | 50 |  | 0 | 0 | 15 | 0 | 7 | 1578.2 |

## Notable Examples

### empty_response

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/hrm8k/gemma-4-E4B-gguf-q8/samples_hrm8k_gsm8k_2026-05-06T02-45-26.939493.jsonl` doc `2` target `70000`

```text

```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/hrm8k/gemma-4-E4B-gguf-q8/samples_hrm8k_gsm8k_2026-05-06T02-45-26.939493.jsonl` doc `37` target `2`

```text

```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/hrm8k/gemma-4-E4B-gguf-q8/samples_hrm8k_ksm_2026-05-06T02-45-26.939493.jsonl` doc `0` target `\frac{1}{6}`

```text

```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/hrm8k/gemma-4-E4B-gguf-q8/samples_hrm8k_ksm_2026-05-06T02-45-26.939493.jsonl` doc `2` target `81`

```text

```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/hrm8k/gemma-4-E4B-gguf-q8/samples_hrm8k_ksm_2026-05-06T02-45-26.939493.jsonl` doc `3` target `\frac{x-3}{4}`

```text

```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/hrm8k/gemma-4-E4B-gguf-q8/samples_hrm8k_ksm_2026-05-06T02-45-26.939493.jsonl` doc `4` target `\frac{e\pi}{e-1}`

```text

```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/hrm8k/gemma-4-E4B-gguf-q8/samples_hrm8k_ksm_2026-05-06T02-45-26.939493.jsonl` doc `6` target `32\pi`

```text

```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/hrm8k/gemma-4-E4B-gguf-q8/samples_hrm8k_ksm_2026-05-06T02-45-26.939493.jsonl` doc `7` target `8`

```text

```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/hrm8k/gemma-4-E4B-gguf-q8/samples_hrm8k_ksm_2026-05-06T02-45-26.939493.jsonl` doc `8` target `\frac{32}{15}\pi`

```text

```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/hrm8k/gemma-4-E4B-gguf-q8/samples_hrm8k_ksm_2026-05-06T02-45-26.939493.jsonl` doc `9` target `\frac{7}{4}`

```text

```

### possible_extraction_miss

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/gsm8k_cot_zeroshot/gemma-4-E4B-gguf-q8/samples_gsm8k_cot_zeroshot_2026-05-06T01-45-34.806046.jsonl` doc `0` target `Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
#### 18`

```text
**A: Let's think step by step.**

**1. Calculate the total number of eggs used (consumed):**
*   Breakfast eggs: 3
*   Muffin eggs: 4
*   Total used: $3 + 4 = 7$ eggs

**2. Calculate the number of eggs remaining to be sold:**
*   Total eggs laid: 16
*   Eggs used: 7
*   Eggs remaining: $16 - 7 = 9$ eggs

**3. Calculate the total money made at the farmers' market:**
*   Eggs sold: 9
*   Price per egg: $2
*   Total earnings: $9 \times \$2 = \$18$

**Answer:** Janet makes **$18** every day at the f
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/gsm8k_cot_zeroshot/gemma-4-E4B-gguf-q8/samples_gsm8k_cot_zeroshot_2026-05-06T01-45-34.806046.jsonl` doc `1` target `It takes 2/2=<<2/2=1>>1 bolt of white fiber
So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric
#### 3`

```text
**Step 1: Determine the amount of blue fiber.**
The robe requires 2 bolts of blue fiber.

**Step 2: Determine the amount of white fiber.**
The robe requires half the amount of blue fiber.
White fiber = 1/2 * 2 bolts = 1 bolt.

**Step 3: Calculate the total number of bolts.**
Total bolts = Blue fiber + White fiber
Total bolts = 2 bolts + 1 bolt = 3 bolts.

**Answer:** It takes a total of **3 bolts**.
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/gsm8k_cot_zeroshot/gemma-4-E4B-gguf-q8/samples_gsm8k_cot_zeroshot_2026-05-06T01-45-34.806046.jsonl` doc `2` target `The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000
He increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000
So the new value of the house i`

```text
**Step 1: Calculate the Total Investment (Cost)**
Josh's total investment is the purchase price plus the repair costs.
*   Purchase Price: $80,000
*   Repair Costs: $50,000
*   **Total Investment:** $80,000 + $50,000 = $130,000

**Step 2: Calculate the Increase in Value (The Gain)**
The value increased by 150% of the original purchase price ($80,000).
*   Increase Amount: $80,000 * 150% (or 1.5)
*   **Increase Amount:** $120,000

**Step 3: Calculate the Final Selling Price (Revenue)**
The final 
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/gsm8k_cot_zeroshot/gemma-4-E4B-gguf-q8/samples_gsm8k_cot_zeroshot_2026-05-06T01-45-34.806046.jsonl` doc `3` target `He sprints 3*3=<<3*3=9>>9 times
So he runs 9*60=<<9*60=540>>540 meters
#### 540`

```text
**Step 1: Calculate the total number of sprints James runs in one week.**
*   He runs 3 sprints per session.
*   He runs 3 sessions per week.
*   $3 \text{ sprints/session} \times 3 \text{ sessions/week} = 9 \text{ total sprints per week}$

**Step 2: Calculate the total distance run.**
*   He runs 60 meters per sprint.
*   He runs 9 sprints per week.
*   $9 \text{ sprints} \times 60 \text{ meters/sprint} = 540 \text{ meters}$

**Answer:** James runs a total of **540 meters** a week.
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/gsm8k_cot_zeroshot/gemma-4-E4B-gguf-q8/samples_gsm8k_cot_zeroshot_2026-05-06T01-45-34.806046.jsonl` doc `4` target `If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3*20=<<3*20=60>>60 cups of feed per day.
If she feeds the flock 15 cups of feed in the morning, and 25 cups in the aft`

```text
**A: Let's think step by step.**

**Step 1: Calculate the total amount of feed Wendi needs for the entire day.**
*   Wendi has 20 chickens.
*   Each chicken needs 3 cups of feed per day.
*   Total daily feed = $20 \text{ chickens} \times 3 \text{ cups/chicken} = 60 \text{ cups}$.

**Step 2: Calculate the total amount of feed Wendi has already given.**
*   Morning meal: 15 cups.
*   Afternoon meal: 25 cups.
*   Total feed given so far = $15 \text{ cups} + 25 \text{ cups} = 40 \text{ cups}$.

**St
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/gsm8k_cot_zeroshot/gemma-4-E4B-gguf-q8/samples_gsm8k_cot_zeroshot_2026-05-06T01-45-34.806046.jsonl` doc `6` target `If Seattle has 20 sheep, Charleston has 4 * 20 sheep = <<20*4=80>>80 sheep
Toulouse has twice as many sheep as Charleston, which is 2 * 80 sheep = <<2*80=160>>160 sheep
Together, the three has 20 shee`

```text
**Step 1: Find the number of sheep in Charleston.**
*   Charleston has 4 times as many sheep as Seattle.
*   Seattle has 20 sheep.
*   Charleston = 4 * 20 = 80 sheep.

**Step 2: Find the number of sheep in Toulouse.**
*   Toulouse has twice as many sheep as Charleston.
*   Charleston has 80 sheep.
*   Toulouse = 2 * 80 = 160 sheep.

**Step 3: Calculate the total number of sheep.**
*   Total = Toulouse + Charleston + Seattle
*   Total = 160 + 80 + 20
*   Total = 260 sheep.

**Answer:** Toulouse, 
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/gsm8k_cot_zeroshot/gemma-4-E4B-gguf-q8/samples_gsm8k_cot_zeroshot_2026-05-06T01-45-34.806046.jsonl` doc `10` target `The number of downloads of the program in the second month increased to 3*60 = <<3*60=180>>180
In the first two months, the total number of downloads of the program was 180+60 = <<180+60=240>>240
In t`

```text
**Month 1:**
*   Downloads = 60

**Month 2:**
*   The downloads were three times the first month.
*   $60 \times 3 = 180$
*   Downloads in Month 2 = 180

**Month 3:**
*   The downloads were reduced by 30% from the second month (180).
*   *Calculate the reduction:* $180 \times 0.30 = 54$
*   *Calculate the final downloads:* $180 - 54 = 126$
*   Downloads in Month 3 = 126

**Total Downloads:**
*   Total = Month 1 + Month 2 + Month 3
*   Total = $60 + 180 + 126$
*   Total = 366

**Answer:** The pro
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/gsm8k_cot_zeroshot/gemma-4-E4B-gguf-q8/samples_gsm8k_cot_zeroshot_2026-05-06T01-45-34.806046.jsonl` doc `11` target `The total charge for the doughnuts was 3 x $68 = $<<3*68=204>>204.
The total charge for the mini cupcakes was 2 x $80 = $<<2*80=160>>160.
The total charge for the mini cheesecakes was 6 x $55 = $<<6*5`

```text
**Step 1: Calculate the cost of the donuts.**
*   3 dozen donuts $\times$ $68 per dozen = $204

**Step 2: Calculate the cost of the mini cupcakes.**
*   2 dozen mini cupcakes $\times$ $80 per dozen = $160

**Step 3: Calculate the cost of the mini cheesecakes.**
*   6 dozen mini cheesecakes $\times$ $55 per dozen = $330

**Step 4: Calculate the total cost.**
*   $204 (donuts) + $160 (cupcakes) + $330 (cheesecakes) = $694

**Answer:** The total cost was **$694**.
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/gsm8k_cot_zeroshot/gemma-4-E4B-gguf-q8/samples_gsm8k_cot_zeroshot_2026-05-06T01-45-34.806046.jsonl` doc `12` target `He makes $10.5 selling lemons each year because 7 x 1.5 = <<7*1.5=10.5>>10.5
He earns $7.5 each year from the lemon tree because 10.5 - 3 = <<10.5-3=7.5>>7.5
It will take 12 years to earn enough to pa`

```text
**Step 1: Calculate the annual revenue.**
Carlos grows 7 lemons per year, and each lemon sells for $1.50.
Annual Revenue = $7 \times \$1.50 = \$10.50$

**Step 2: Calculate the annual net profit (after operating costs).**
It costs $3.00 per year to water and feed the tree.
Annual Net Profit = Annual Revenue - Annual Operating Cost
Annual Net Profit = $\$10.50 - \$3.00 = \$7.50$

**Step 3: Determine the initial investment (the amount that needs to be recovered).**
Initial Cost = $90.00

**Step 4: 
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/gsm8k_cot_zeroshot/gemma-4-E4B-gguf-q8/samples_gsm8k_cot_zeroshot_2026-05-06T01-45-34.806046.jsonl` doc `14` target `There are 20 x 20/100 = <<20*20/100=4>>4 students who enrolled in contemporary dance.
So, 20 - 4 = <<20-4=16>>16 students are enrolled in either jazz or hip-hop dance.
There are 16 x 25/100 = <<16*25/`

```text
**Step 1: Calculate the number of students in Contemporary Dance.**
*   Total students = 20
*   Contemporary enrollment = 20% of 20
*   $0.20 \times 20 = 4$ students

**Step 2: Calculate the number of remaining students.**
*   Remaining students = Total students - Contemporary students
*   $20 - 4 = 16$ students

**Step 3: Calculate the number of students in Jazz Dance.**
*   Jazz enrollment = 25% of the *remaining* students (16)
*   $0.25 \times 16 = 4$ students

**Step 4: Calculate the number 
```

### very_long_response

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/leaderboard_ifeval/gemma-4-E4B-gguf-q8/samples_leaderboard_ifeval_2026-05-06T02-54-36.410078.jsonl` doc `9` target `0`

```text
Subject: Invitation to Strategic Planning Summit Q3 2024

Dear Esteemed Participants

We are delighted to extend a formal invitation to a critical strategic planning summit. This meeting represents a pivotal moment for our organization a chance to align our collective vision and chart a definitive course for the coming fiscal quarters. Your expertise your insights and your dedication are absolutely essential to the success of this endeavor. We believe that the collective wisdom present in this g
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/leaderboard_ifeval/gemma-4-E4B-gguf-q8/samples_leaderboard_ifeval_2026-05-06T02-54-36.410078.jsonl` doc `10` target `0`

```text
# The Art of the Sway: Why a Hammock is the Ultimate Wellness Tool

In a world that demands constant motion—emails pinging, deadlines looming, and the relentless hum of modern life—we often treat rest as a luxury, something we squeeze in when we have time. But what if I told you that true, restorative rest doesn't require a complicated routine or an expensive spa day? What if the answer is a simple piece of woven fabric suspended between two trees?

The hammock, often relegated to the realm of v
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/leaderboard_ifeval/gemma-4-E4B-gguf-q8/samples_leaderboard_ifeval_2026-05-06T02-54-36.410078.jsonl` doc `11` target `0`

```text
```json
{
  "advertisement_campaign": {
    "campaign_name": "CloudCuddle Comfort Revolution",
    "product_details": {
      "product_name": "CloudCuddle Diapers",
      "category": "Baby Care / Diapers",
      "unique_selling_proposition": "Unparalleled softness and breathability, designed to eliminate irritation and maximize baby comfort.",
      "key_features": [
        "Ultra-soft, hypoallergenic material",
        "Advanced moisture-wicking technology",
        "Breathable design for opti
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/leaderboard_ifeval/gemma-4-E4B-gguf-q8/samples_leaderboard_ifeval_2026-05-06T02-54-36.410078.jsonl` doc `13` target `0`

```text
the social network: a cold, brilliant dissection of modern ambition

david fincher’s 2010 masterpiece, "the social network," is far more than a mere biopic about the creation of facebook. it is a razor-sharp, often brutal examination of ambition, betrayal, and the corrosive nature of modern digital life. it’s a film that doesn't just tell a story; it dissects the anatomy of a cultural phenomenon, presenting the birth of a global empire through the lens of legal drama and personal wreckage.

the 
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/leaderboard_ifeval/gemma-4-E4B-gguf-q8/samples_leaderboard_ifeval_2026-05-06T02-54-36.410078.jsonl` doc `24` target `0`

```text
# Phil Larkin: Professional Clown & Performance Artist

***

### Section 1: Professional Summary & Contact

**Phil Larkin**
(555) 123-4567 | phil.larkin@email.com | [LinkedIn Profile URL/Portfolio Link]

**Summary:**
Highly energetic, versatile, and seasoned professional clown with 15+ years of experience delivering high-impact, family-friendly, and theatrical entertainment. Specializing in physical comedy, character development, and audience engagement, I excel at transforming any event—from co
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/leaderboard_ifeval/gemma-4-E4B-gguf-q8/samples_leaderboard_ifeval_2026-05-06T02-54-36.410078.jsonl` doc `31` target `0`

```text
ਇਹ ਰੂਬਰਿਕ (ਮਾਪਦੰਡ) ਸੌਫਟਵੇਅਰ ਇੰਜੀਨੀਅਰ ਦੇ ਤਕਨੀਕੀ ਹੁਨਰਾਂ ਦਾ ਮੁਲਾਂਕਣ ਕਰਨ ਲਈ ਤਿਆਰ ਕੀਤਾ ਗਿਆ ਹੈ। ਇਸ ਵਿੱਚ ਕੋਈ ਹੋਰ ਭਾਸ਼ਾ ਵਰਤੀ ਨਹੀਂ ਗਈ ਹੈ।

***

# ਤਕਨੀਕੀ ਹੁਨਰਾਂ ਦਾ ਮੁਲਾਂਕਣ ਮਾਪਦੰਡ
**(Technical Skills Evaluation Rubric)**

**ਮੁਲਾਂਕਣ ਦਾ ਉਦੇਸ਼:** ਇੰਜੀਨੀਅਰ ਦੀ ਕੋਡਿੰਗ, ਸਮੱਸਿਆ ਹੱਲ ਕਰਨ ਦੀ ਸਮਰੱਥਾ, ਅਤੇ ਸਿਸਟਮ ਡਿਜ਼ਾਈਨ ਦੀ ਗੁਣਵੱਤਾ ਦਾ ਅਧਿਐਨ ਕਰਨਾ।

**ਮੁਲਾਂਕਣ ਦੇ ਪੱਧਰ:**
1. **ਬਹੁਤ ਵਧੀਆ (Bahut Vadhia):** (5 ਅੰਕ) - ਉੱਚਤਮ ਮਿਆਰ, ਬਿਨਾਂ ਕਿਸੇ ਕਮੀ ਦੇ।
2. **ਵਧੀਆ (Vadhia):** (4 ਅੰਕ) - ਮਿਆਰ ਪੂਰਾ ਕਰਦਾ ਹੈ, ਛੋਟੀਆਂ ਕਮੀਆਂ ਹੋ ਸਕਦੀਆਂ ਹਨ।
3. 
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/leaderboard_ifeval/gemma-4-E4B-gguf-q8/samples_leaderboard_ifeval_2026-05-06T02-54-36.410078.jsonl` doc `36` target `0`

```text
The Tang Dynasty (618–907 CE) represents one of the most glorious and expansive eras in Chinese history, a time defined by extraordinary cultural achievement, vast territorial reach, and complex cycles of both profound **peace** and brutal **war**.

The early years of the Tang saw the consolidation of power following the collapse of the Sui Dynasty. While the initial campaigns were marked by necessary **war** to subdue rebellious warlords and unify the fractured realm, the dynasty quickly establ
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/leaderboard_ifeval/gemma-4-E4B-gguf-q8/samples_leaderboard_ifeval_2026-05-06T02-54-36.410078.jsonl` doc `37` target `0`

```text
"The sentiments held by the Croatian and Serbian nations towards the Austrian Empire (the Habsburg Monarchy) at the close of the 19th century were profoundly complex, contradictory, and highly volatile. They were not monolithic; rather, they represented a spectrum of political ideologies, social classes, and cultural aspirations, ranging from pragmatic loyalty and constitutional reformism to fervent, uncompromising nationalism and outright anti-imperial resistance. The late 19th century was a pe
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/leaderboard_ifeval/gemma-4-E4B-gguf-q8/samples_leaderboard_ifeval_2026-05-06T02-54-36.410078.jsonl` doc `40` target `0`

```text
"## From ARPANET to TikTok: How the Internet Changed Everything (And Why You Should Care)

Hey everyone! 👋

Take a second and look around. Seriously. Look at your phone, your laptop, the smart speaker in your room. Chances are, whatever you’re looking at is connected to the internet. It’s the invisible force that lets you stream music, chat with friends across the globe, binge-watch the latest series, and even order pizza without leaving your couch.

But have you ever stopped to think about wher
```

- `results/eval_scores/20260505T163642Z_gemma-4-E4B-gguf-q8_full/leaderboard_ifeval/gemma-4-E4B-gguf-q8/samples_leaderboard_ifeval_2026-05-06T02-54-36.410078.jsonl` doc `49` target `0`

```text
(The setting is imagined: a formal address, perhaps from the Oval Office, delivered with measured cadence and profound conviction.)

My fellow citizen,

I have been presented with a simple object. A small, unassuming piece of currency. A dime. In the grand tapestry of our nation’s economy, a dime is a whisper—a fraction of a cent in the vast ledger of global commerce. It is easily overlooked, easily dismissed as mere pocket change, a trifle of negligible value. But I tell you today, as your Pres
```
