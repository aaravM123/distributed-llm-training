import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("results/benchmark_results.csv")

plt.figure()
plt.plot(df["mode"], df["elapsed"])
plt.title("Epoch Time vs Mode")
plt.ylabel("Epoch Time (s)")
plt.savefig("plots/epoch_time_vs_mode.png")


plt.figure()
plt.bar(df["mode"], df["throughput"])
plt.title("Throughput vs Mode")
plt.ylabel("Samples/s")
plt.savefig("plots/throughput_vs_mode.png")