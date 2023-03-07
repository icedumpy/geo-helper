import helper
import pandas as pd
import time
# =============================================================================
# Check icepackage.df_tools.set_index_for_loc
# =============================================================================
df = pd.DataFrame(["A12", "2B", "Cc1"], index=["ice", "dumpy", "ice"])
df['index'] = df.index

# Normal
t = time.process_time()
for i in range(1000):
    df[df['index']=="ice"]
print(f"Time usage (normal): {(time.process_time() - t):.4f} s")

# .loc unsorted
t = time.process_time()
for i in range(1000):
    df.loc["ice"]
print(f"Time usage (.loc unsorted): {(time.process_time() - t):.4f} s")

# .loc sorted
df = helper.df_tools.set_index_for_loc(df, "index")
t = time.process_time()
for i in range(1000):
    df.loc["ice"]
print(f"Time usage (.loc sorted): {(time.process_time() - t):.4f} s")
#%%
