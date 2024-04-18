import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/trajectories.csv')
df_filtered = df[df['id'] == 55]

print(df_filtered)

plt.plot(df_filtered.index, df_filtered['l_speed'])

plt.xlabel('Frame Number')
plt.ylabel('Speed (pixels/sec)')
plt.title('Speed Over Existence of Salmon ID 55')

plt.show()
