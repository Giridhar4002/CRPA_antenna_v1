# app.py

import matplotlib.pyplot as plt
import numpy as np

class Dashboard:
    def __init__(self):
        self.elevation_data = self.load_elevation_data()

    def load_elevation_data(self):
        # Load the elevation data
        return np.random.rand(10)

    def plot_elevation(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Setting up adaptive y-limits
        y_max = self.elevation_data.max() * 1.1
        y_min = self.elevation_data.min() * 0.9
        ax.set_ylim(y_min, y_max)

        # Unified visual styling
        ax.plot(self.elevation_data, linewidth=2, label='Elevation Data', color='blue')
        ax.grid(alpha=0.5)
        ax.set_title('Elevation Profile', fontsize=16, fontweight='bold')
        ax.set_xlabel('Distance (km)', fontsize=14)
        ax.set_ylabel('Elevation (m)', fontsize=14)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', labelsize=12)

        # Improve layout spacing
        plt.tight_layout()
        plt.annotate('Elevation Pattern', xy=(5, self.elevation_data[5]), xytext=(6, self.elevation_data[5]+0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, fontweight='medium')

        plt.show()

if __name__ == '__main__':
    dashboard = Dashboard()
    dashboard.plot_elevation()