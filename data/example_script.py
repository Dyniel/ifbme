
import pandas as pd
import numpy as np

# Load training data
train_df = pd.read_csv("trainData.csv")

# Load validation data
val_df = pd.read_csv("valData.csv")

# Very basic model: predict constant values (for example purposes only)
DTestimation = np.random.randint(0, 2, size=len(val_df))
LSestimation = np.random.randint(1, 15, size=len(val_df))

# Save outputs
np.savetxt("DTestimation.csv", DTestimation, delimiter=",", fmt="%d")
np.savetxt("LSestimation.csv", LSestimation, delimiter=",", fmt="%d")
