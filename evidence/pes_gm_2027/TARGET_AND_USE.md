# Target and use

This note states what the forecast is and what it is not. It does not estimate trading
returns, settlement cost, or dispatch value.

## Who would use the forecast

A one-hour-ahead forecast of the forthcoming ERCOT hourly maximum hub-average
real-time price is a descriptive input that a short-horizon analyst, a load-serving
entity, or a researcher could consult. No user, position, or decision rule is
identified in the study. The paper therefore reports accuracy, tails, bias, and
event detection as separate objects.

## What the target measures

The target at issue time t (UTC hour start) is the maximum of the HB_HUBAVG
15-minute settlement-point prices whose interval starts are t, t+15, t+30, and
t+45 minutes. Incomplete hours keep the maximum of the intervals that exist.
HB_HUBAVG is a hub average, not a zonal or nodal settlement price and not a
volume-weighted exposure.

## Why this is not settlement cost or exposure

ERCOT real-time energy is settled at 15-minute intervals at settlement points.
An hourly maximum of a hub average:

- discards the other three intervals in a complete hour;
- ignores quantity, so it is not a mark-to-market P&L;
- ignores day-ahead positions, congestion, and ancillary-service settlements;
- is not the price paid by any specific QSE or retail customer.

A $200/MWh threshold is an inherited code constant. Its scientific rationale is
[MISSING] beyond that constant. Spike precision and recall are descriptive.

## What the February 2021 diagnostic is

The 2021 evaluation is a chronological stress check with a pre-2021 fit. Official
FERC/NERC, ERCOT, and PUC timestamps define onset, emergency, and recovery
sub-windows after those cuts were committed and before the sub-window metrics
were computed. The diagnostic does not claim a causal mechanism, a resilience
result, or a recommended operating policy.

## What is not estimated

No trading strategy, no dispatch recommendation, no scarcity adder, and no
consumer-bill calculation is produced. Accuracy on ordinary hours and errors on
scarcity hours are reported separately so they are not substituted for each other.
