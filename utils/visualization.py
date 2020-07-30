import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_jasons_lineplot(
    x,
    y, 
    x_label,
    y_label,
    title,
    output_png_path,
    ):
    
    if x == None:
        x = range(1, len(y) + 1)

    _, ax = plt.subplots()
    plt.plot(x, y, linewidth=1)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=400)
    plt.clf()

    print(f"plot saved at {output_png_path}")