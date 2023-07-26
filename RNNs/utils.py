import matplotlib.pyplot as plt

def plot_xy(title: str, x: list, y: list) -> None:
    plt.plot(x, y)
    plt.title(title)
    plt.show()