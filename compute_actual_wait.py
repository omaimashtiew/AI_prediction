import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 1. Load the data
status = pd.read_csv("fencestatus.csv")
fences = pd.read_csv("fences.csv")

# 2. Clean columns and dates
status.columns = status.columns.str.strip()
fences.columns = fences.columns.str.strip()
status['message_time'] = pd.to_datetime(status['message_time'])

# 3. Merge tables to get fence names
merged = pd.merge(status, fences, left_on='fence_id', right_on='id')
merged.sort_values(by=['fence_id', 'message_time'], inplace=True)

# 4. Calculate waiting periods
waits = []

for fence_id, group in merged.groupby('fence_id'):
    group = group.sort_values('message_time')
    for i in range(len(group) - 1):
        current = group.iloc[i]
        next_row = group.iloc[i + 1]

        if current['status'] in ['closed', 'sever_traffic_jam'] and next_row['status'] == 'open':
            wait_time = abs((next_row['message_time'] - current['message_time']).total_seconds() / 60)
            wait_start = current['message_time']
            if 6 <= wait_start.hour < 24:  # Only include daytime hours
                waits.append({
                    'fence_id': fence_id,
                    'fence_name': current['name'],
                    'start_time': wait_start,
                    'wait_minutes': wait_time
                })

# 5. Convert results to DataFrame
wait_df = pd.DataFrame(waits)

# 6. Prepare the line plot with points
fig, ax = plt.subplots(figsize=(15, 8))

# Define date range for plot
start_date = datetime(2025, 3, 20)
end_date = datetime(2025, 5, 4)

# Group data by fence and calculate daily averages
for fence_name, group in wait_df.groupby('fence_name'):
    # Aggregate by date (daily average)
    daily_avg = group.groupby(group['start_time'].dt.date)['wait_minutes'].mean().reset_index()
    daily_avg['start_time'] = pd.to_datetime(daily_avg['start_time'])
    
    # Plot the line with points
    ax.plot(daily_avg['start_time'], 
            daily_avg['wait_minutes'], 
            marker='o', 
            linestyle='-',
            markersize=8,
            linewidth=2,
            label=fence_name,
            alpha=0.8)

# Formatting
ax.set_xlim([start_date, end_date + timedelta(days=1)])
ax.set_ylim(bottom=0)  # Start y-axis at 0

# Titles and labels
ax.set_title("Average Daily Wait Times at Checkpoints (March 20 - May 4, 2025)", 
             fontsize=16, pad=20)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Average Wait Time (minutes)", fontsize=12)

# Date formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))  # Major ticks on Mondays
ax.xaxis.set_minor_locator(mdates.DayLocator())  # Minor ticks for each day

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Grid and legend
plt.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend(title="Checkpoint Name", 
          bbox_to_anchor=(1.05, 1), 
          loc='upper left',
          framealpha=1)

# Adjust layout to prevent clipping
plt.tight_layout()

# Add some padding at the top
plt.subplots_adjust(top=0.9)

# Show the plot
plt.show()