import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

df = pd.read_csv("actual_waits.csv")

# Prepare the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Scatter plot with color gradient based on waiting time
sc = ax.scatter(
    df['start_time'], 
    df['wait_minutes'], 
    c=df['wait_minutes'], 
    cmap='viridis',
    alpha=0.7,
    edgecolor='black'
)

# Titles and labels
ax.set_title("Distribution of Waiting Times at Checkpoints", fontsize=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Waiting Time (minutes)", fontsize=12)

# Format x-axis for date display
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

# Add color bar
cbar = plt.colorbar(sc)
cbar.set_label('Waiting Time (minutes)', fontsize=12)

# Format y-axis to show hours and minutes
def format_minutes_to_hours(x, pos):
    hours = int(x) // 60
    minutes = int(x) % 60
    return f"{hours}h {minutes}m"

ax.yaxis.set_major_formatter(plt.FuncFormatter(format_minutes_to_hours))

plt.tight_layout()
plt.grid(True)
plt.show()