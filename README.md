# ARX Scorer
Absolute-Relative XGBoost Scorer (Blueprint)

## How does it work?
I won't use fancy jargons.

### Relative Difficulty Scoring

1) Using 100+ maps, each with 10 replays, I train the machine to 
   understand what patterns made players do worse and better
2) These effects are normalized, that means hard maps are weighed
   similarly to easier maps. This makes comparison fair.
3) Using these relationships, I can predict maps' difficulty
   relatively.

(2) is a major assumption.

### Absolute Difficulty Scoring

Interestingly enough, if you don't normalize relative scoring,
you get a decent estimator of difficulty.

Hence, this is the same, but we skip (2).

## Template

If you want to use the predicted value, you can use ``model.predict``

This will return a `(windows, features=8)` array.
Each feature is an "independent" estimator for the combined final result
in ``predict_and_plot_agg``.

### Absolute Difficulty

```python
import numpy as np
import matplotlib.pyplot as plt
from input import Preprocessing

from model import XGBoostModel
np.random.seed(0)
map = Preprocessing(4).features_from_path("path/to/map.osu")
model = XGBoostModel.load(4, "mm_mm")
plt.figure(figsize=(6,2.5))
model.predict_and_plot_agg(map)
plt.show()
```

### Relative Difficulty
```python
import numpy as np
import matplotlib.pyplot as plt
from input import Preprocessing

from model import XGBoostModel
np.random.seed(0)
map = Preprocessing(4).features_from_path("path/to/map.osu")
model = XGBoostModel.load(4, "mm_mm_s")
plt.figure(figsize=(6,2.5))
model.predict_and_plot_agg(map)
plt.show()
```

## Results

Here is a sample result.

https://twitter.com/dev_evening/status/1397587243682844677

## Status

The status of this is on hold until I receive a reply on the
osu!mania data dump.

I did think about having many people to help me retrieve replays,
but why waste 10000 hours of mundane work when I can ask the
developer to do an SQL query?

