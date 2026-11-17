# Autoresearch log

Branch `autoresearch/nov12`. Metric: strict-fill PnL, days -1 and 0 summed (V15 = $1,741.09; standard $2,151.88).
Evaluate with `python -m research.autoresearch_eval trader_v16.py` from `starter_kit/starter_kit`.
Target: strict total >= $4,000. Each entry: hypothesis, command, result, what it taught.

## Exp 1 - passive Kit exit (keep, +$7.32 strict, $1,748.41)
Hypothesis: exiting the directional Kit hold with a passive quote in its last 20 ticks saves spread vs the timed sweep.
Command: `python -m research.autoresearch_eval trader_v16.py`. Frozen rule from `kit_passive_exit_batch` (window 20, edge 0).
Result: strict -1 +$0.84, day 0 +$6.48; standard +$8.69; slip1 +$0.85/+$6.47; guardrails pass.
Taught: reproduces the earlier research exactly, so the evaluator is trustworthy; the gain is small and it is batch material only.

## Metric change (evidence-based)
The server marks the two options at model value using the true daily vol. Both options quote at ONE constant implied vol per day
(day -1: 4.42, day 0: 2.25) versus realised pills vol 3.26 / 3.16 - so mid-marked local PnL hides the option edge. Platform day-0
weather PnL ($59.71) vs local mid-marked ($4.65) matches model marking (+$51 at sigma = realised). The evaluator now marks options
at model value (full-day realised vol; 0.9x and 1.1x reported as sensitivity). V15 control = $1,864.54 on this basis.

## Exp 2 - warrants toward model value (keep, +$130.35, $1,994.88)
Hypothesis: bots quote warrants at a day-constant implied vol that differs from true vol; buy when model > ask, sell when bid > model, hold to close.
Details: realised vol from V15's 20-tick samples (>= 50 samples); session date (time to expiry 3.0 vs 2.0 days) inferred from which choice leaves implied vol flat tick 10 -> 600.
Result: day -1 +$93.3, day 0 +$29.7; holds at 0.9x and 1.1x vol; guardrails pass.
Taught: the options mispricing is real and persistent but capped by position limits (100 warrants x ~30c). Needs the server to mark at true vol = realised; day-0 weather platform result supports this.

## Exp 3 - policy trading changes (discard)
Policy min-samples 200 -> 50 lost $160 (noisy early vol estimate flips the policy position); using inferred expiry date or removing the 0.85 vol shrink changed nothing.
Taught: the policy position is sensitive to early vol noise; leave V15's policy logic alone.

## Exp 4 - policy-implied latent pills price (keep, +$31.73, $2,026.61)
Finding: the policy mid, inverted at its constant implied vol (EWMA-calibrated online), pins the pills price (day -1 deviation std 0.7) and the pills book mid reverts toward it:
next-tick corr 0.52 day -1, 0.24-0.35 day 0 (slope 0.2-0.95 depending on policy sensitivity). One-tick R2 for pills rises 0.09 -> 0.29 (day -1 validation half).
Rule: alpha += sens/(sens+2.5) * (S_implied - pills_mid), clip +-4, added to the Pills reservation price; 100-tick warmup.
Result: day -1 +$17.3, day 0 +$14.4 (strict, model-marked); standard +$21; guardrails pass.
Taught: bots quote derivatives off the latent price; derivative quotes are a cleaner price source than the product's own book. Only used for Pills quotes so far; Kit gap and Pills aggression are next.

## Exp 5-6 - discards
5: policy signal weight/clip/scale sweep is flat or worse (noise 1.0: -8.3, 6.0: -20.0, x0.7: -9.5, x1.3: -1.0, clip 8: 0). Taught: exp 4 sits at a local optimum.
6: Gas alpha from the Kit residual (Kit - Glass - Pills - Gas) has no predictive power (R2 0.00 at 1/5/20 ticks; residual std 30). Taught: Gas mid is close to a random walk at tick scale; Kit book noise is too large to infer Gas from.

## Exp 7 - Pills inventory skew 0.5 (keep, +$33.81, $2,060.42)
Hypothesis: with a better pills fair-price estimate the quote needs less inventory skew.
Sweep (strict model-marked total): skew 1.0 2026.6, 0.75 2043.9, 0.5 2060.4, 0.25 2070.8, 0.0 2062.2 - smooth and monotone, positive on both days; standard total falls 2428.8 -> 2411.8 at 0.5.
Chose 0.5 rather than the strict peak 0.25 to limit the standard-fill give-back.
Taught: part of V15's skew was compensating for a noisy fair value.

## Exp 8 - directional Kit SIZE 25 (keep, +$25.59, $2,086.01)
Sweep of SIZE: 10 -> 2045.3, 15 -> 2060.4, 20 -> 2069.4, 25 -> 2086.0 (limit); positive on both days, standard also up. Hurdle 15 fails guardrails, 45 loses 24, DELAY 2 loses 3.
Taught: the Kit gap is an OU process (std 25-30, autocorr half-life ~100 ticks, spread 10.8); the limit was binding, so use all of it.

## Exp 9 - Gas Masks parameter sweep (discard)
improve=1: strict +38.8 but standard -164.1; skew 0.5 -8.3; skew 2 -51.3; size 30 0; size 10 -20.7; edge 1 -25; edge 3 -115.8.
Taught: the only strict gain comes from inside quotes whose fills depend on the fill model, so it is not trusted. Gas quotes are tuned.

## Exp 10 - passive band cycling on the policy (discard, -$80.1)
Hypothesis: the policy is 30c wide with ~10k units/day of flow, so cycling 5 units around the favourable limit earns ~28c per round trip.
Result: mid-marked PnL rose slightly but model-marked fell: average inventory sits below the limit at the close (loses the model edge) and day -1 lost $52.
Taught: terminal inventory edge (~$5 per unit) outweighs the spread capture; keep options at the limit.

## Exp 11 - simplification (keep, same PnL)
Warrants traded identically with the inferred, correct and wrong expiry date (all hit +/-100 in the same direction), so the inference was removed. Fixed 2.5 days is used for the model value.

## Findings that did not become experiments
* No cross-product lead-lag with the latent pills price (corr < 0.04 for Gas/Glass/Kit).
* Platform export holds only a PnL curve (no fills), so it cannot recalibrate the fill model.

## Platform check of V16 (day 0, submitted by user)
V15 $1,149.10 -> V16 $1,211.94 (+$62.84): warrants +$30.86 (local est +$29.7), Pills +$30.05 (+$33), Kit +$4.10, Glass +$2.54, Gas -$4.71, policies unchanged. Zero errors, drawdown $25.75.
Calibration: platform day 0 ~= strict + 0.55 * (standard - strict), options model-marked. The evaluator now prints this `blend` (V15 control uses the same formula).
Leaderboard (day-0 backtest profit): $6,014.52, $4,514.40, $1,924.75, $1,831.69. Day-0 only; final ranking uses two hidden days.

## Exp 12 - warrant-implied pills signal (discard)
Adding a warrant-quote deviation (delta-weighted) on top of the policy signal: blend -1.5 (scale 0.5), -2.8 (scale 1.0). Taught: the policy already carries the information; warrant delta is too small.

## Exp 13 - Pills crosses the spread on the latent-price alpha (keep, +$28.2 strict, $2,114.21)
Rule: for Pills only, if floor(reservation - edge - 1) >= best ask, buy at the ask (mirror for sells); otherwise quote as before.
Margin sweep (strict / standard): 0: +11.3 / +55.5; 1: +28.2 / +58.1; 2: +6.7 / +33.0; 3: 0 / +1.3. Both fill models agree; day -1 +$24.5, day 0 +$3.7 (strict).
Taught: when the policy-implied price is clearly away from the book, taking beats quoting; most of the gain is on days where the policy is near the money (strong signal).
Not adopted: re-tuning the policy signal weight (noise 1.5: blend +7.5; clip 3: +3.7) - small and tuned on the same two days.

## Platform check of latest V16 (Pills crossing) - day 0 $1,226.51
+$14.57 vs first V16 (Pills +$10.91, Gas +$4.18). Local blend estimate $1,216.7. Features now go in trader_v17.py (V16 frozen, tag `v16-uploaded`).

## Exp 15 - passive Kit entry (discard, V17)
Rest a passive order at the inside for 2/3/6 ticks at the directional entry, then sweep if unfilled. Strict -16.5 / -20.8 / -11.4; blend -17 / -12 / -5.
Taught: the timed aggressive entry beats saving the spread; the signal decays faster than a passive order fills, matching the earlier capped-entry result.

## Exp 16 - Gas Masks crossing on its own alpha (discard)
Margin 0: strict -92.5; margin 1: -13.9. Taught: Gas alpha (weak ridge on a near random walk) is too noisy to cross on; confirms R2 ~ 0.007.

## Exp 17 - Glass crossing (keep, V17, +$64.88 strict, $2,179.09)
Generalised the Pills crossing rule to a per-product margin table `TAKE_EXTRA`. For Glass, cross when floor(reservation - edge - margin) >= ask (mirror for sells).
Margin sweep (strict / blend vs V16): +1: +6.8 / n/a; 0: +48.8 / +42.7; -1: +64.9 / +60.1; -2: +36.9 / +35.4. Margin -1 means cross whenever the reservation price reaches the quote. Day -1 +$31.9, day 0 +$33.0; standard +$56.1.
Taught: Glass's pinned fair value (~1000) makes its reservation price reliable enough to take rather than only quote; Gas's is not.

## Exp 18 - Glass quote parameters with crossing (discard)
skew 0.5: -0.5; skew 2: -40.1; clip 3: +0.3; weight 1.5: -11.9; weight 0.7: -5.3 (strict vs V17). Taught: Glass quoting is at its optimum; the crossing was the missing piece.

## V17 status
Strict model-marked two-day total $2,179.09 (V16 $2,114.21, V15 $1,864.54). Day-0 blend estimate for the platform: about $1,258 (V16 platform $1,226.51 + local blend gain of about $31).

## Exp 19 - basket thresholds (discard)
BASE_EDGE 2 and 6 give identical PnL (the volatility term dominates); VOL_EDGE_MULTIPLIER 1.5: -4.5, 3.0: +3.9 strict but standard -21. Taught: basket arb is insensitive here.

## Exp 20 - V18 simplification (keep)
Removed the Glass hybrid rule: PnL unchanged (-$0.16) because crossing now covers it. V18 = V17 minus that rule; V17 not edited.

## Exp 21 - Glass long-run anchor (discard)
Pull Glass reservation toward a slow EWMA of its mid (Glass is pinned near 1000): strict -3.4 to +0.8, standard +21 to +24, blend +11 to +13. Not adopted: strict-neutral and non-monotone in weight, so the standard gain is probably fill-model sensitivity rather than edge.

## Exp 22 - periodicity scan (no strategy)
Trade counts and |mid change| by tick modulo 2, 3, 4, 5, 10, 20 for Gas, Pills, Glass, Kit on both days: chi2/df about 1 and relative std of |dmid| below 0.08, i.e. no bot refresh cadence to exploit.
Plateau note: after exps 19-22 the remaining parameter sweeps (basket thresholds, Glass params, Gas params, policy-signal weights) are flat; new gains need a new mechanism.

## Platform check of V18 (day 0) - $1,273.47
+$46.96 vs V16 (Glass +$49.44 as predicted from Glass crossing, Gas +$3.81, Pills +$0.53, Kit -$6.82). Estimated $1,258; calibration within +$15.
User direction after V18: stop small steps, rebuild for large gaps; avoid strategies that cannot be verified on hidden data; probe submissions allowed; batch changes.

## Exp 24 - long-horizon Gas/Pills position rules (discard) - research/long_horizon_scan.py
Variance ratios hinted Gas mean-reverts and Pills trends over 500-2000 ticks, but target-position rules around an EWMA anchor (half-life 300/1000/3000, gain 0.25-1, both signs) lose money for every configuration after touch costs (gross edge about $20-50/day, turnover cost $100-300). Against a 200-shuffle random-walk null the rules are mostly mid-percentile. Taught: ten 2000-tick windows over two days made the variance ratios look stronger than they are.

## Exp 23 - policy passive market maker (discard)
Prototype showed +$175 strict but only 2 fills: the day-0/-1 gain came from entering the policy position late, not from spread capture. Cause (new structural finding): option flow is ONE-DIRECTIONAL each day - policy and warrant bot takers only SELL on day -1 (0 buys / 10,268 units) and only BUY on day 0 (10,127 / 0). The "$1,450/day capture ceiling" assumed two-way flow and is not available. Taught: bot flow in the options trades toward true value, so a maker is always adverse-selected against the model.

## Capacity table (inside quotes, one tick, markout vs mid, both days)
Edge per filled unit and whether the flow is two-way: Glass 2.5 (two-way), Pills 1.45, Gas 3.3 (decays to ~1.9-3.3 at 20 ticks), Kit 4.2-4.6 (decays to ~2 at 20 ticks), warrants 1.4 and policy 14.5 (one-way only).
V18 resting units per day: Glass ~13k (flow ~18.5k), Pills 10-12k (17-18k), Gas only ~1.1k (13-14k flow), Kit ~0.

## Exp 25 - V19 Gas inside quotes with strong inventory skew (keep) - $2,355.93 strict (+$177.0 vs V18)
Gas improve 0 -> 1, micro 2.0 -> 0, skew 1 -> 8, size 20 -> 35. Gas-only strict PnL: V18 83.8 / 124.0 -> 102-108 / 222-247 across a plateau of skew 6-15, edge 2-3, size 20-50 (both days up). Inside quotes fill ahead of bots, so strict is the right fill model here; platform Gas has tracked strict (V15 $143 vs strict $101 / standard $430; V18 $147 vs strict $124 / standard $427).
Without strong skew the same quotes were day-dependent (day -1 about -$2, day 0 +$407): inventory trend exposure, not edge. Day -1 +$40.6, day 0 +$139.8 for the whole portfolio; standard-fill total falls $117 because V18's join quotes were credited for at-price prints that the platform has not paid.
Expected platform day 0: about $1,400 (V18 $1,273 + ~$125).

## Exp 26 - Kit fair-anchored maker overlay (discard)
Offline sim of a fair-anchored Kit maker (fair = component mids + slow mean gap, inside quotes, limit 25) earned $70-110 day -1 and $175-190 day 0, but as an overlay it lost $30-108 on day -1 and gained $24-76 on day 0 (net -63..+17). It competes with the directional Kit hold for the same 25-unit limit.

## Evaluator
Replays now run with GC disabled; collection pauses read as slow ticks and tripped the guard.

## Platform check of V19 (trader_v19.py, user called it "iteration 18") - day 0 $1,388.34
+$114.87 vs V18 ($1,273.47); estimate was ~$1,400. Gas $273.34 (V18 $146.70; local strict estimate ~$247), Pills $298.45 (+2.1), Glass $505.12 (-0.05), Kit $220.86 (-13.83), warrants/policies unchanged. Drawdown $18.34 (V18 $25.75); zero errors.
Taught: Gas inside quotes are paid at about strict fills or a little better; the strict-fill model is the right one for Gas. Kit lost $13.83 for an unknown reason (possibly Gas inventory interplay with basket hedges).

## Exp 28 - Kit/Gas interplay diagnosis (no change; trader_v20.py created as a copy of V19)
Question: why did platform Kit fall $13.83 (V18 $234.69 -> V19 $220.86) when only the Gas config changed?
Local replay does NOT reproduce it: Kit PnL V19 vs V18 is +$4.6 (day -1) and +$7.8 (day 0); Kit units traded are unchanged (912 -> 931, 1,188 -> 1,106 aggressive); Gas inventory is lower in V19 (ticks at |pos| >= 45: 10% -> 5%, 14% -> 4%); repair cost and hedge imbalance are the same. So there is no capacity crowding.
Platform vs local for V19-V18 (curve comparison): platform +$114.5 vs local strict +$135.8 (84% realised). The $21 shortfall builds early (curve points 10-60, i.e. ticks ~1,000-3,000) and then stays constant; product split on the platform is Gas +126.6 (local +129), Kit -13.8 (local +7.8), Pills +2.1 (+3.6), Glass -0.05 (-4.3).
Taught: the Kit shortfall is a ~$20 live-versus-local effect seen in one early window, not a Gas interaction I can reproduce; chasing it blind is worth at most ~$14. Correction to earlier suggestion: Pills already quotes inside (improve 1) and fills 60-70% of flow, so it is also near its ceiling.

## Exp 29 - Kit rebuild (V20): fair-anchored maker + basket arb + directional - $2,378.82 strict (+$22.89 vs V19)
Split of Kit's local PnL: basket arbitrage is the main earner (turning it off costs $175 over two days), the directional trade is small (turning it off costs only $48).
Maker design: fair = component mids + slow EWMA of the Kit gap; quote bid = min(bid+1, floor(fair - edge - skew*inv/25)), ask mirrored; inventory tracked separately from the basket and directional overlays so the basket repair logic never sees it; stops quoting at tick 9900.
Findings: (1) an uncapped maker with the directional trade removed fills 1.6k-2.3k units but cuts basket aggressive units from ~1,000 to ~270 (it eats the 25-unit limit) -> day -1 -$60 on Kit; (2) capping maker inventory at 10-15 units protects the basket. Both-day-positive configuration: directional SIZE 25 + maker cap 10, size 10, edge 0: day -1 +$7.0, day 0 +$15.9 (strict), blend +$25.3. Other configurations were mixed (day -1 -13 to -60, day 0 +15 to +45).
Taught: Kit's flow is large but its 25-unit limit is the constraint, and basket arbitrage uses it better than quoting does. Rebuilding Kit as one controller yields +$23, not a large jump.
Evaluator: tick-time guard now uses mean tick time (single-tick OS jitter caused false failures; the harness shows <= 6 ms max).

## Platform check of V21 (V20 trading + telemetry) - day 0 $1,483.25 (+$94.91 vs V19)
Kit $319.69 (+$98.83; local estimate +$8 strict / +$41 on this day), Gas $265.92 (-7.42), Pills $300.63 (+2.18), Glass $506.44 (+1.32), warrants/policies unchanged; drawdown $23.47; zero errors; max tick 1.69 ms.
Telemetry findings (log cap = 20,000 chars = first 129 ticks only):
 * Platform books equal the CSV books EXACTLY for Glass, Pills, Gas and Kit over all 129 logged ticks while we traded (0 mismatches): no visible bot reaction in the displayed book in that window.
 * Platform fills match the local fill structure but are a little larger on many events (platform events 31, local strict 26, standard 32; quantities differ on 11 events, mostly platform >= local).
 * Per product, platform day-0 PnL for V19 minus local strict: Glass +$99, Pills +$67, Gas +$20, Kit +$3. Local strict is conservative for products quoted inside by $150-190/day.
 * V21-minus-V19 curve: platform rises steadily to +$95; local strict peaks near +$38 mid-day and ends +$16. The Kit maker earns steady spread on the platform; locally its gains partly reverse.
 * Marking Kit at component value instead of Kit mid changes local Kit PnL by only ~$1, so valuation does not explain the Kit gap.
Taught: the platform pays inside quotes more than the strict replay; the Kit maker (cap 10) is worth about $99 live. Use per-product probes, since the platform reports PnL by product.

## Exp 30 - V22: larger Kit maker (cap 20, size 20) - $2,436.45 strict (+$57.63 vs V21)
Local sweep over the maker with the directional trade on (strict totals): cap/size 10/10 2,378.8; 15/15 2,417.2; 20/20 2,436.5 (day -1 +27.9, day 0 +29.7); 25/25 2,424.7; skew 4 2,396.8; skew 16 2,441.8; edge 1 2,433.0. A flat plateau, both days up. Chosen cap 20, size 20, edge 0, skew 8.
Probes considered and rejected from V22: Pills improve 2 (local -$45, both days down) and Glass improve 2 (local -$200); nothing in the V21 logs shows that a better price attracts extra flow.
Telemetry now logs the market tape plus our orders and fills (first ~268 ticks fit the 20,000-character log cap).
Expected platform day 0: V21 $1,483.25 + $30 local (the platform paid 2.4x the local Kit gain last time) = about $1,515-1,560.

## Platform check of V22 (trader_v22.py) - day 0 $1,545.31 (+$62.06 vs V21)
Glass $513.60 (+7.16), Pills $302.41 (+1.78), Gas $280.53 (+14.61), Kit $358.21 (+38.52), warrants/policies unchanged. Drawdown $26.12, zero errors, max tick 1.71 ms. Local estimate was +$30; the platform paid about 2x again.
Tape telemetry (first ~246 ticks; the log shows only the tape left after our fills):
 * Our fills are taken out of the historical taker flow: CSV prints that our inside quotes took (e.g. Kit 2617x8 at tick 16, matched by our 8-unit fill at 2616) are missing from the platform tape. Consistent with the standard fill model for prints at or through our price.
 * But the platform fills us MORE than the CSV tape contains: in the window the CSV tape volume consumed was 746 units while our fills totalled 904, and 43 of 108 ticks do not balance (e.g. Gas tick 52: CSV tape 15, we filled 17).
 * Seven ticks show platform prints that are absent from the CSV tape (bots trading among themselves differently). So bots do react to our presence in the flow, even though the displayed books stayed identical to the CSV.
Taught: live flow is richer than the fixed tape, which is why inside-quoted products pay 1.5-2.5x their local strict gain. This favours size and inside quotes, but the local replay cannot rank quote aggressiveness, so aggressiveness changes need platform probes.

## V23 - alternating-setting probe (not a candidate for the final upload)
Purpose: learn whether live flow rewards quote aggressiveness or size, which the fixed-tape replay cannot rank. V22 logic, except every tick the Glass and Pills quotes alternate between improve 1 and improve 2 (tick parity) and Glass size alternates between 50 and 20 (tick//2 parity), so both settings face the same market.
Local cost: strict $2,305.15 vs V22 $2,436.45 (day 0 -$67.5); platform effect unknown.
Telemetry: from tick 50, one short line per tick: "tick|v<improve flag><size flag>|mG..|mP.. (tape)|F.. (our fills)"; the 20,000-character log covers ticks 50-1,403.
Analysis plan: a fill printed at tick t+1 belongs to the variant active at tick t. Compare fills per tick, fill size and edge per unit for improve 1 vs 2 and size 50 vs 20; compare tape volume consumed with our fills; compare the platform Glass and Pills PnL with V22 ($513.60 / $302.41).

## Platform check of V23 (alternating probe) - day 0 $1,452.76 (-$92.55 vs V22)
Glass $434.96 (-78.64), Pills $288.05 (-14.36), Gas $280.38 (-0.15), Kit $358.81 (+0.60), warrants/policies unchanged; drawdown $26.12; zero errors. Half the ticks quoted two cents inside, so about half of the loss is the price of that setting; local replay predicted a loss of this sign (day 0 -$67.5 total).
Fill analysis (logged ticks 50-887, fill at tick t+1 attributed to the setting at tick t):
 * Glass: improve 2 filled 2.57 units/tick vs improve 1 2.04 (+26%), but each unit earns about 1c less of a 2.5c edge, so revenue per tick falls ~24%.
 * Pills: improve 2 filled FEWER units (1.92 vs 2.42 per tick).
 * Glass size 50 vs 20: no effect (improve 1: 2.10 vs 1.98; improve 2: 2.47 vs 2.67 units/tick).
Taught: live flow is only mildly price-elastic and not size-elastic; V22's quote placement (one cent inside, large size) is near the optimum. The extra fills beyond the CSV tape are not explained by quote aggressiveness or size. Closed lever.

## V24 - telemetry-free V22 (final-candidate cleanup)
Identical trading logic and local PnL to V22 ($2,436.45 strict, $2,538.21 standard); no print output; 33,709 bytes. Upload to confirm the platform reproduces $1,545.31.

## Hidden-edge search, step 1 - executable basket arbitrage with full depth (research/basket_depth_audit.py)
A hedged policy (buy Kit / sell components when cheap, reverse when rich, sweep up to 3 levels, target +-25 baskets, min edge per basket e) locks in: e=0: $50 / $87; e=3: $72 / $128; e=6: $94 / $157; e=10: $139 / $175; e=15: $138 / $170 (day -1 / day 0). Our current basket logic earns about $66 / $109. Gain available about $50/day, not the $500/day the hindsight oracle shows. Closed as a large edge.

## Hidden-edge search, step 2 - V25 reaction probe (not a final candidate)
Question: do live bots react to our AGGRESSIVE orders (market impact)? The displayed books matched the CSV exactly while we only quoted passively, so impact is the remaining untested channel and the likeliest source of an edge the fixed replay cannot show.
Design: V24 trading plus ten scheduled small aggressive orders (ticks 80-440; Glass, Pills, Gas and Kit; buys and sells; sizes 10-30), each followed by a 12-tick log of all four books and our fills. research/reaction_probe_analysis.py compares those books with the CSV and reports the mid shift in the push direction at +1, +3, +6, +12 ticks and the spill-over into other products.
Local cost: $2,437.59 strict (V24 $2,436.45), so the probe is free in the replay; real cost on the platform is a few dollars of spread.

## Platform check of V25 (reaction probe) - day 0 $1,528.32 (-$16.99 vs V22/V24 expected $1,545.31)
Glass $510.26, Pills $301.71, Gas $269.08, Kit $356.71, warrants $30.86, policies $59.71; drawdown $26.12; zero errors. The $17 shortfall is the cost of the ten probe orders (spread plus the inventory they leave behind).
Reaction analysis (research/reaction_probe_analysis.py): across all ten events (Glass, Pills, Gas, Kit; buys and sells; 10-30 units) the displayed books for all four products over the 12 ticks after each event match the CSV exactly: 0 of 520 books differ, mid shift 0.0 at +1/+3/+6/+12 ticks, spill-over 0.0. Our fills at each event (price and quantity, including multi-level sweeps) match the local strict replay exactly.
Taught: the displayed market is an exogenous replay that does not react to our orders, aggressive or passive; there is no market-impact edge. Whatever the top teams earn must come from information inside the replay (hindsight of the day-0 path, or real predictability), not from moving bots.

## Hidden-day proxy - time-reversed practice days (research/perturbed_days.py)
Linear products only (Glass, Pills, Gas, Kit), strict fills, mid-marked, both days summed: V15 1,728 -> 1,509 reversed; V19 2,096 -> 1,727; V24 2,177 -> 1,901. V24 keeps +$392 over V15 on reversed days (+$449 on the original), and the Glass, Gas and Kit gains hold (Gas 400 -> 482, Kit 473 -> 509).
Fragility: Pills falls from $483 to $114 under V24 on reversed days, and returns to $403 with the policy signal switched off. Reversal makes the quoted time-to-expiry increase while the signal assumes it decreases, so this is probably a calendar artifact (hidden days keep the forward calendar), but it shows the signal depends on that assumption. A faster implied-vol calibration (EWMA 0.01) restores reversed Pills to $365 and keeps day 0 unchanged, at a local cost of $35 on day -1 (strict total $2,401.56 vs $2,436.45). Not adopted: the calendar is the same on hidden days and day 0 was confirmed on the platform; noted as the main known risk.

## Hidden-edge search, step 3 - structure scan
Spectral and cross-day scan of Gas, Pills and Kit mids: power is concentrated in the lowest frequencies (84-95% in the lowest five), no sharp periodicity, and the detrended paths correlate inconsistently across days (Gas +0.43, Pills -0.67, Kit +0.77). No deterministic or seasonal component to trade.

## Option-regime stress (research/option_regime_stress.py) and V26
The option quotes were regenerated from the real pills path at implied vols 1.5, 2.25, 3.2 (about realised), 4.42 and 6.0 and V24 was replayed with options marked at model value. With a large gap between implied and realised vol the logic earns well (implied 6.0: +$154 warrants / +$148 policies on day -1; +$84 / +$169 on day 0; implied 4.42 reproduces the real day -1 gain of about $93 / $81). With implied close to realised (3.2) it leaks: day 0 warrants -$61.4, and day -1 implied 2.25 warrants -$17.6.
Fix (trader_v26.py): WARRANT_EDGE 10 -> 20 cents. Real-day results are identical to V24 ($2,436.45 strict, $2,538.21 standard); the stress leaks disappear (day 0 implied 3.2: $0, day -1 implied 2.25: $0) and the large-gap gains are kept (day 0 implied 4.42: +$33.0, unchanged). A 35-cent threshold gives up $1.8 on the real days, so 20 is the setting.
Taught: the warrant gain is positive in every regime with a clear vol gap; the remaining downside is bounded by the policy leg (about -$16 near the realised vol).

## V27 - Kit controller packaged as a standalone trader (trader_v27.py)
Source: research/deep_kit_controller.py and research/reports/deep_edges/kit_selected_parameters.json from another research session in this repo (research/DEEP_HIDDEN_EDGE_RESEARCH.md). It replaces all Kit sleeves (basket arbitrage, directional, maker) with a dynamic-programming controller: AR(1) fitted on the first 5,000 ticks of day -1 (persistence 0.99222, innovation sd 4.17c), value table over gap (-150..150 step 3) x inventory (-25,-15,0,15,25), execution priced through the three book levels, passive quotes from the marginal continuation value. Components, options and warrants stay as V26.
Standalone result (research/autoresearch_eval): strict $2,641.12 (day -1 $1,254.08, day 0 $1,387.04) vs V26 $2,436.45: +$56.0 / +$148.7. Standard $2,773.45 (+$52 / +$183). All guardrails pass; 44,325 bytes. Day 0 is an out-of-sample check (parameters fitted on early day -1 only); the late-day-1 selection window lost $9.43 in the source research.
Time-reversed hidden-day proxy (linear products, strict): V27 2,147 vs V24 1,901 (+$246; originals 2,382 vs 2,177, +$205). Kit $647 vs $509 reversed, $651 vs $473 original. The Kit gain carries to a different path.
Known costs: strict drawdown rises (about $43 vs $37 on day -1); no platform result yet.

## Platform check of V27 (Kit controller) - day 0 $1,584.26 (+$38.95 vs V22's $1,545.31)
Glass $530.33 (+16.73), Pills $307.34 (+4.93), Gas $259.56 (-20.97), Kit $396.47 (+38.26), warrants/policies unchanged; drawdown $20.72 (lowest so far; V22 $26.12); zero errors; max tick 1.56 ms.
Local expectation was +$149 on day 0 (Kit +$118), so the platform paid about 26% of the local gain.
Why: for V27 the platform Kit PnL ($396) equals the local strict Kit PnL ($393), whereas V22's Kit maker earned $358 on the platform vs about $275 locally. The controller's local gain is mostly aggressive execution (local = platform), and it dropped the V22 maker's passive-fill premium.
Tests run after the result:
 * V28 (controller with the V22-style fair-anchored passive quotes): local strict $2,539.31, $102 below V27 on both days; not pursued (trader_v28.py kept for reference).
 * Swap-fit test: the Kit table fitted on early day 0 (persistence 0.98841, sd 3.52) instead of early day -1 (0.99222, 4.17) gives only +$4.1 on day -1 and +$59.9 on day 0 versus V26, against +$56.0 / +$148.6 for the shipped table. The shipped gain depends on the fitted parameters; the out-of-sample range is +$4 to +$149, so expect well below the in-sample $205 on hidden days.
 * V29 (trader_v29.py): table fitted on pooled parameters (persistence 0.99031, sd 3.84): strict $2,575.52 (+$38.4 / +$100.6 vs V26), standard $2,666.13, guardrails pass (a one-off 161 ms timing spike on this machine was jitter). Less parameter-dependent but not platform-tested.
Last-upload rule: the file graded is the last upload before the lock (Nov 16, 8 p.m. Central); V27 is currently the last upload.

## Hidden-edge search, step 4 - why the platform fills more than the CSV tape (V22 log, platform orders replayed through the CSV matcher)
Method: feed the orders the platform actually received (V22 telemetry, ticks 0-245) through a standalone strict matcher on the CSV books/tape and compare with the platform's own fills (no path divergence).
 * Every tape fill the matcher predicts is also on the platform. The platform's extra units come in two kinds.
 * Kind 1 (Glass 29, Pills 47, Gas 18 units): the CSV tape consumed the ENTIRE touch level (e.g. 18 of 18 displayed) and the platform gave us more (21, 22, 15 vs 12...). The bot taker's size exceeds the displayed touch and its limit sits at the touch price, so the unfilled remainder is invisible in the tape but lands on us when we quote one cent inside.
 * Kind 2 (Pills 113, Kit 42 of 50 units, Gas 3, Glass 0): fills with NO tape trade at all, e.g. Pills sells at 1127 / 1126 inside a 1123/1128 book. Hidden takers whose limits lie strictly inside the visible spread; no trade happens in the bot-only world, so the CSV never records them.
 * V23 showed this flow is only mildly price-elastic (+26% units for one more cent inside, Pills none), so a one-cent improvement over the touch already collects almost all of it. Flow is bounded; the maker edge per unit (~2.5c) is set by the bots' spread. That caps market making near today's level.
 * Scale check: limit 50 x a ~300c Gas move = $150 per product per day even with perfect timing; the hindsight oracle ($6.2k) needs every swing. Leaders near $6k on day 0 are consistent with a path-specific fit.
Open (cheap) question: does the live platform supply anything the replay lacks? trader_v30.py = V27 logic (identical local PnL 245,809) plus a tiny log of observations.plain_values, listings, limits, attribute list and any named counterparties. Costs nothing in PnL.

## V31 - neural-net alpha test (platform test requested by the user)
Design: tiny tanh MLP ensembles (3 nets, 4-8 hidden units, L2, early stopping) per product (Glass, Pills, Gas), trained on practice day -1 ONLY, so platform day 0 is a true out-of-sample test. Inputs: mid changes over 1/2/5/10/20 ticks, book imbalance (L1, L3), spread, Kit gap, level-2 offset. Horizon/size chosen on the second half of day -1 (Glass h=20, Pills h=1, Gas h=1). The net replaces the ridge alpha in V27; everything else is V27 (trader_v31.py, 53,478 bytes; research/nn_alpha_train.py, build_nn_trader.py, nn_variant_eval.py).
Forecast R^2 (out of sample): Glass 0.43 (day 0), Pills 0.085 (day 0; 0.09-0.12 reversed), Gas 0.067 (day 0). Real but small, and the ridge already captures most of it.
Strict linear-product PnL, V27 -> replace-all: day -1 (in-sample) 1080 -> 1047, day 0 (held out) 1302 -> 1358, reversed -1 1051 -> 1123, reversed 0 1096 -> 1125. Per product: Pills replace gives most of the gain (+51 on day 0), Glass/Gas replace are ~0.
Evaluator: strictM total 2,664.46 vs V27 2,641.12 (+23.3), standard +37.6, guardrails PASS, mean tick 0.34 ms.
Taught (before platform): a net adds about +$25 locally and loses on its own training day; no large jump, consistent with the earlier tree result. Platform day 0 is the real out-of-sample check.

## Platform check of V31 (neural-net alpha) - day 0 $1,682.04 (+$97.78 vs V27 $1,584.26)
Glass $545.11 (+14.78), Pills $386.96 (+79.62), Gas $262.93 (+3.37), Kit $396.47 (0), warrants $30.86, policies $59.71; drawdown $17.24 (V27 $20.72); zero errors; max tick 2.26 ms, mean 0.99 ms. Local day-0 prediction for the Pills gain was +$51 strict (platform pays ~1.5x on inside-quoted products), so the live result matches the local replay's direction and rough size.
Swap test (research/nn_alpha_train.py with NN_TRAIN_DAYS=0): nets trained on day 0 give Pills out-of-sample forecast R^2 0.13 on day -1, but the PnL effect is day dependent: Pills-replace changes strict linear PnL by -34 on day -1 (also -37 when trained on day -1), +56 on day 0, +107 / +55 on the two reversed days. Glass replace ~0; Gas replace negative locally (-11 orig, -19/-31 reversed).
Flow check: signed tape volume has ~0 correlation with the next 1-5 tick mid change for all three products (|corr| <= 0.08), so no flow features were added.
Taught: the NN signal is real but small and uneven across days; day 0 is its best day, so do not extrapolate +$98 to hidden sessions (a fairer expectation is +$20-50 per session). V31 is the only version with an out-of-sample platform gain (trained on day -1, scored on day 0) and is the current best final candidate.

## NN follow-ups after V31 (research/nn_feature_probe.py)
 * Richer inputs (all six quote levels' prices and volumes, change since last tick, distance from slow EWMAs): mean out-of-sample R^2 (train one day, test the other day and both reversed days) Pills h=1 0.1076 -> 0.1116, Gas h=1 0.0557 -> 0.0532; h=5 equally flat. The base ten inputs already hold the predictable part; no new net was built.
 * Scaling the Pills net alpha (0.7, 1, 1.5, 2, 3): strict linear PnL on the original days 2,388 / 2,396 / 2,390 / 2,378 / 2,370 (flat to falling), reversed days rising to 2,367 at scale 2. No reliable gain; scale 1 kept.
 * Replacing Glass and Gas alphas as well (V31) vs Pills only: 2,405 vs 2,396 on the original days, 2,247 vs 2,310 reversed: no meaningful difference.
Closed: the neural alpha is saturated with the available book features.

## V32 - V31 trading logic + the V30 introspection log (final candidate and probe in one)
Trading is identical to V31 (local 248,143 cents both days; same drawdown); it only prints a one-time line with observations, listings, limits and attributes, plus any named counterparties. Upload to (a) confirm the platform reproduces $1,682.04 and (b) see whether the platform exposes observations or counterparty names. 56,150 bytes. If the result reproduces V31 it can be the last upload.

## Platform check of V32 (V31 + introspection log) - day 0 $1,682.04, exact reproduction of V31
PnL 168,203.91 cents, every product and the whole 215-point PnL curve identical to V31; drawdown $17.24; zero errors; max tick 1.67 ms, mean 0.85 ms. The platform is deterministic and the added prints do not change behaviour.
Introspection: observations.plain_values is empty ({}), listings and position limits equal the local replay (Glass/Pills/Gas 50, Kit 25, warrants 100, policies 10; denomination CENTS), state attributes are the nine documented ones, market-trade buyer/seller are blank, own trades are tagged SUBMISSION. Nothing on the live platform supplies extra signals or counterparty names. Closed: no hidden information channel.
V32 is now the final candidate: it reproduces V31 exactly and carries the log harmlessly.

## Final-day robustness audit of V32 (Nov 16)
- Reversed-day strict linear PnL (two days summed): V27 2146.8, V32 2247.3 (original days 2381.6 vs 2404.9). V32 >= V27 on both.
- Option regime stress (implied vol 1.5-6.0 on both days): option PnL never negative except -16.0 on one cell; no tail loss.
- Book-corruption fuzz (research/fuzz_books.py): thin books 0 errors; one-sided/empty/dropped ticks raise per-tick exceptions (harness counts them, PnL continues, 0 limit violations). Real data has 0 missing sides in 120k rows.
- Glass fair value is an EMA, not a hardcoded 1000. No day-0-fitted level constants found beyond product specs (strikes 1200/800).
- Conclusion: no change; V32 stays final. A both-day NN retrain cannot be validated out of sample, so it was not built.
