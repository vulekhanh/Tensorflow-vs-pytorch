import time
import pandas as pd

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        return time.time() - self.start_time


def save_results(framework, total_time, final_acc):
    df = pd.DataFrame({
        "Framework": [framework],
        "Total Time (s)": [total_time],
        "Final Accuracy": [final_acc]
    })

    filename = f"results_{framework.lower()}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved results to {filename}")
