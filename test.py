import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
file_path = 'ptz_data1.csv'  # 실제 파일 경로로 변경해주세요.
data = pd.read_csv(file_path)

sampled_data = data.iloc[::5]

plt.figure(figsize=(10, 6))
plt.scatter(sampled_data['x'], sampled_data['y'], alpha=0.5)

for i, point in sampled_data.iterrows():
    plt.text(point['x'], point['y'], ' ' + point['Timestamp'], fontsize=6, color='gray')

plt.title('Scatter Plot of x and y Coordinates with Timestamps')
plt.xlabel('x Coordinate')
plt.ylabel('y Coordinate')
plt.grid(True)
save_path = 'scatter_plot_with_timestamps.png'
plt.savefig(save_path)
plt.show()
