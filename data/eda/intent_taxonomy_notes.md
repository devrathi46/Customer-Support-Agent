# Intent taxonomy: design notes

Built from `src/eda_intents.py` (TF-IDF + KMeans over an 8,000-message
sample of AppleSupport customer texts, 16 clusters) — see console output
captured 2026-09-17.

## Key observation

This dataset's AppleSupport traffic is entirely from the iOS 11 launch
window (Oct-Dec 2017). The overwhelming majority of clusters were shades of
the same handful of issues: battery drain/slowdown after updating, app
freezing/crashing, the well-known "I" -> "I️" autocorrect glitch, and
WiFi/Bluetooth problems. Distinct minority clusters covered account/
purchase/billing (Apple Music, App Store refunds, spam emails), repair/
warranty requests, and general how-to questions.

**This is a real limitation to flag in the report's "misleading headline
number" section**: a taxonomy and agent tuned on this window will
over-fit to "iOS 11 bug complaint" patterns and may not generalize to
AppleSupport traffic from other time periods with a different issue mix.

## Why 7 intents, not more

Several TF-IDF clusters were the *same* underlying issue restated
different ways (e.g. clusters for "the I glitch", "question mark box",
"letter bug" are all the same autocorrect defect) — merging these avoids
an inflated, redundant taxonomy that would fragment training/eval data
per class. Conversely, `performance_battery` and `update_installation_issue`
were kept separate from general `software_bug_glitch` because they call for
different response content (battery diagnostics steps vs. update
troubleshooting steps vs. generic bug acknowledgment) even though they
share surface vocabulary ("update", "iOS 11").

## Why no dedicated "positive feedback" class

"Thanks" appeared as a frequent term, but almost always inside sarcastic/
frustrated messages ("thanks for the garbage update 🙄"), not genuine
positive feedback. A dedicated class would likely be near-empty and noisy;
folded into `general_inquiry_other` instead.
