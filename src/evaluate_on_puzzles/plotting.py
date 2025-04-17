import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

CMAP = mcolors.LinearSegmentedColormap.from_list("WhiteRed", ["white", "red"])

def plot_bar(result_df,out_path = None,puzzle_result_name= None,choice = 'elo',no_plot = False):
    """
    choice = 'elo' or 'puzzle_len'
    """
    if no_plot == False:
        assert out_path is not None
        assert puzzle_result_name is not None

    df = result_df.copy()
    total_counts = []
    correct_counts = []

    if choice == 'elo':
        sorted_x_label = sorted(df.columns, key=lambda x: int(x.split('-')[0]))
        xlabel = 'Puzzle Rating (Elo) - Count'
        title_name = 'Rating Interval'
        save_name = '_Elo_bar.png'
    elif choice == 'puzzle_len':
        df = df.T
        sorted_x_label = sorted(df.columns)   #, key=lambda x: int(x.split('-')[0]))
        xlabel = 'Puzzle Length - Count'
        title_name = 'Puzzle Length'
        save_name = '_puzzlelen_bar.png'
    else:
        raise ValueError("Invalid selection")


    for name in sorted_x_label:
        col_totals = df[name].apply(lambda cell: cell[1])
        col_corrects = df[name].apply(lambda cell: cell[0])
        correct, total = col_corrects.sum(), col_totals.sum()
        total_counts.append(total)
        correct_counts.append(correct)
    percentages = [
        (corr / tot) if tot != 0 else 0.0
        for corr, tot in zip(correct_counts, total_counts)
    ]
    overall_acc = np.sum(correct_counts) / np.sum(total_counts)

    x_labels = [f"{name}\n{tot}" for name, tot in zip(sorted_x_label, total_counts)]
    x_positions = np.arange(len(sorted_x_label))

    if no_plot:
        return x_labels, percentages, overall_acc

    plt.figure(figsize=(15, 6))
    plt.bar(x_positions, percentages)
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy (%)')
    plt.ylim([0.0, 1.0])
    plt.yticks(np.arange(0.0, 1.05, 0.05))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plot_title = f"Percentage of Correct Results per {title_name}\n {out_path.split('/')[-1]}\n Acc = {overall_acc:.4f}"
    plt.title(plot_title)
    plt.xticks(x_positions, x_labels)
    plt.tight_layout()

    plt.savefig(os.path.join(out_path, puzzle_result_name + save_name))
    # plt.show()
    plt.close()

def plot_heatmap(result_df, out_path, puzzle_result_name):
    df = result_df.copy()
    total_counts = 0.0
    correct_counts = 0.0
    for idx in df.index:
        for col in df.columns:
            cell_val = df.loc[idx, col]
            corr, tot = cell_val  # unpack the two values
            total_counts += tot
            correct_counts += corr
            if tot == 0:
                pct = np.nan  # mark as missing so that it will appear in dark grey
            else:
                pct = corr / tot
            df.loc[idx, col] = pct
    overall_acc = correct_counts / total_counts
    # print(overall_acc)
    df = df.astype(float)
    df = df.sort_index()

    # Convert the DataFrame values to a masked array where np.nan values are masked
    masked_array = np.ma.masked_invalid(df.values)

    # Make a copy of the colormap and set masked values ("bad") to grey
    cmap_mod = plt.get_cmap(CMAP).copy()
    cmap_mod.set_bad(color='lightgrey')

    plt.figure(figsize=(10, 6))
    plt.imshow(masked_array, aspect='auto', cmap=cmap_mod, vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(label="% Correct")
    plt.xticks(ticks=np.arange(len(df.columns)), labels=df.columns, rotation=45)
    plt.yticks(ticks=np.arange(len(df.index)), labels=df.index)
    plt.xlabel("Puzzle Range")
    plt.ylabel("Puzzle Length")
    plot_title = f"Heatmap of Correct Results Percentage\n {out_path.split('/')[-1]}\n Acc = {overall_acc:.4f}"
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, puzzle_result_name + '_heatmap.png'))
    # plt.show()
    plt.close()

def plot_puzzle_results(result_df,out_path,puzzle_result_name):
    plot_heatmap(result_df, out_path, puzzle_result_name)
    plot_bar(result_df,out_path,puzzle_result_name,choice = 'elo')
    plot_bar(result_df, out_path, puzzle_result_name, choice='puzzle_len')



