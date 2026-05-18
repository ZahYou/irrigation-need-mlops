import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style, init
from pathlib import Path
from IPython.display import display

init(autoreset=True)

_HERE = Path(__file__).parent


class EDA:

    # Color palette — consistent across all plots
    COLORS = {
        'Train': '#3cb371',   # green
        'Test':  '#e74c3c',   # red
        'Orig':  '#3498db',   # blue
    }

    TRAIN_PATH    = str(_HERE / "data" / "train.csv")
    TEST_PATH     = str(_HERE / "data" / "test.csv")
    ORIGINAL_PATH = str(_HERE / "data" / "irrigation_prediction.csv")

    def __init__(self):
        self.train = pd.read_csv(self.TRAIN_PATH, index_col="id")
        self.test  = pd.read_csv(self.TEST_PATH,  index_col="id")
        self.orig  = pd.read_csv(self.ORIGINAL_PATH)
        self.train_orig = pd.concat(
            [self.orig[self.train.columns], self.train], axis=0, ignore_index=True
        )

        self.target = "Irrigation_Need"

        # Encode target variable
        self.train[self.target] = (
            self.train[self.target].map({'Low': 0, 'Medium': 1, 'High': 2}).astype(int)
        )
        self.orig[self.target] = (
            self.orig[self.target].map({'Low': 0, 'Medium': 1, 'High': 2}).astype(int)
        )

        self.cat_features = (self.train
                             .drop(self.target, axis=1)
                             .select_dtypes(include=['object', 'bool'])
                             .columns.tolist())
        self.num_features = (self.train
                             .drop(self.target, axis=1)
                             .select_dtypes(exclude=['object', 'bool'])
                             .columns.tolist())

        self.data_info()
        self.heatmap()
        self.dist_plots_train_test()
        self.dist_plots_orig()
        self.cat_feature_plots()
        self.target_pie()

    # ------------------------------------------------------------------
    def data_info(self):
        table_style = [
            {'selector': 'th:not(.index_name)',
             'props': [('background-color', '#3cb371'),
                       ('color', '#FFFFFF'),
                       ('font-weight', 'bold'),
                       ('border', '1px solid #DCDCDC'),
                       ('text-align', 'center')]},
            {'selector': 'tbody td',
             'props': [('border', '1px solid #DCDCDC'),
                       ('font-weight', 'normal')]}
        ]
        for data, label in zip(
            [self.train, self.test, self.orig],
            ['Train', 'Test', 'Orig']
        ):
            print(Style.BRIGHT + Fore.GREEN + f'\n{label} head\n')
            display(data.head().style.set_table_styles(table_style))

            print(Style.BRIGHT + Fore.GREEN + f'\n{label} info\n' + Style.RESET_ALL)
            display(data.info())

            print(Style.BRIGHT + Fore.GREEN + f'\n{label} describe\n')
            display(data.describe()
                    .drop(index='count',
                          columns=self.target, errors='ignore').T
                    .style.set_table_styles(table_style)
                    .format('{:.3f}'))

            print(Style.BRIGHT + Fore.GREEN + f'\n{label} missing values\n' + Style.RESET_ALL)
            display(data.isna().sum())
        return self

    # ------------------------------------------------------------------
    def heatmap(self):
        print(Style.BRIGHT + Fore.GREEN + '\nCorrelation Heatmap — Train vs Orig\n')

        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                 gridspec_kw={'wspace': 0.4})

        for ax, data, label, cmap in zip(
            axes,
            [self.train, self.orig],
            ['Train (synthetic)', 'Orig (real)'],
            ['Greens', 'Blues']
        ):
            corr = data[self.num_features + [self.target]].corr(method='pearson')
            sns.heatmap(corr, fmt='0.2f', cmap=cmap, square=True,
                        annot=True, linewidths=1, cbar=False, ax=ax)
            ax.set_title(f'Correlation — {label}', fontsize=13, fontweight='bold')

        plt.suptitle('Pearson Correlation Heatmap', fontsize=15, fontweight='bold', y=1.02)
        plt.show()

    # ------------------------------------------------------------------
    def _dist_plot_core(self, df, hue_order, palette, title):
        n = len(self.num_features)
        fig, axes = plt.subplots(
            n, 2,
            figsize=(18, n * 5),
            gridspec_kw={'hspace': 0.4, 'wspace': 0.25,
                         'width_ratios': [0.70, 0.30]}
        )
        if n == 1:
            axes = np.array([axes])

        for i, col in enumerate(self.num_features):
            ax = axes[i, 0]
            sns.kdeplot(
                data=df[[col, 'Source']], x=col, hue='Source',
                palette=palette, ax=ax, linewidth=2,
                hue_order=hue_order,
                common_norm=False
            )
            ax.set_title(f'{col}  —  KDE', fontsize=12, fontweight='bold')
            ax.set(xlabel='', ylabel='Density')
            ax.grid(alpha=0.4)

            ax.relim()
            ax.autoscale_view()
            ymax = ax.get_ylim()[1]
            for src in hue_order:
                color  = palette[src]
                subset = df.loc[df['Source'] == src, col]
                if len(subset):
                    mean_val = subset.mean()
                    ax.axvline(mean_val, color=color,
                               linestyle='--', linewidth=1.2, alpha=0.8)
                    ax.text(mean_val, ymax * 0.92,
                            f'{src}\n{mean_val:.2f}',
                            color=color, fontsize=7,
                            ha='center', va='top')

            ax = axes[i, 1]
            sns.boxplot(
                data=df, y=col, x='Source',
                order=hue_order,
                palette=palette,
                width=0.5, linewidth=1, fliersize=1,
                ax=ax
            )
            ax.set_title(f'{col}  —  Boxplot', fontsize=12, fontweight='bold')
            ax.set(xlabel='', ylabel='')
            ax.tick_params(axis='x', labelsize=10)
            ax.grid(alpha=0.4, axis='y')

        plt.suptitle(title, fontsize=15, fontweight='bold', y=1.005)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    def dist_plots_train_test(self):
        print(Style.BRIGHT + Fore.GREEN +
              '\nDistribution analysis — Train vs Test (shift check)\n')

        df = pd.concat([
            self.train[self.num_features].assign(Source='Train'),
            self.test[self.num_features].assign(Source='Test'),
        ], axis=0, ignore_index=True)

        self._dist_plot_core(
            df=df,
            hue_order=['Train', 'Test'],
            palette={'Train': self.COLORS['Train'], 'Test': self.COLORS['Test']},
            title='Numeric Feature Distributions: Train vs Test'
        )

    # ------------------------------------------------------------------
    def dist_plots_orig(self):
        print(Style.BRIGHT + Fore.GREEN +
              '\nDistribution analysis — Original dataset alone\n')

        df = self.orig[self.num_features].assign(Source='Orig')

        self._dist_plot_core(
            df=df,
            hue_order=['Orig'],
            palette={'Orig': self.COLORS['Orig']},
            title='Numeric Feature Distributions: Original Dataset'
        )

    # ------------------------------------------------------------------
    def cat_feature_plots(self):
        print(Style.BRIGHT + Fore.GREEN +
              '\nCategorical feature distributions — Train / Test / Orig\n')

        n = max(len(self.cat_features), 1)
        fig, axes = plt.subplots(n, 1, figsize=(16, n * 5),
                                 gridspec_kw={'hspace': 0.6})
        if n == 1:
            axes = [axes]

        for i, col in enumerate(self.cat_features):
            ax = axes[i]
            frames = []
            for src, data in [('Train', self.train),
                               ('Test',  self.test),
                               ('Orig',  self.orig)]:
                vc = (data[col].value_counts(normalize=True) * 100).reset_index()
                vc.columns = [col, 'Percentage']
                vc['Source'] = src
                frames.append(vc)

            plot_df = pd.concat(frames, ignore_index=True)

            sns.barplot(
                data=plot_df, x=col, y='Percentage', hue='Source',
                hue_order=['Train', 'Test', 'Orig'],
                palette=self.COLORS, ax=ax, width=0.6
            )
            ax.set_title(f'{col}', fontsize=13, fontweight='bold')
            ax.set(xlabel='', ylabel='Percentage (%)')
            ax.legend(title='Source', fontsize=9)
            ax.grid(alpha=0.4, axis='y')

            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%', fontsize=7, padding=2)

        plt.suptitle('Categorical Feature Distributions: Train / Test / Orig',
                     fontsize=15, fontweight='bold', y=1.005)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    def target_pie(self):
        print(Style.BRIGHT + Fore.GREEN + '\nTarget feature distribution\n')

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, data, label in zip(
            axes,
            [self.train, self.orig],
            ['Train (synthetic)', 'Orig (real)']
        ):
            vc = data[self.target].value_counts()
            ax.pie(vc, labels=vc.index, autopct='%1.2f%%',
                   colors=sns.color_palette('viridis', len(vc)))
            ax.set_title(f'Target Distribution — {label}',
                         fontsize=13, fontweight='bold')

        plt.suptitle(f'{self.target}', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    def target_plot(self):
        print(Style.BRIGHT + Fore.GREEN + '\nTarget feature distribution\n')

        fig, axes = plt.subplots(
            1, 2, figsize=(14, 6),
            gridspec_kw={'hspace': 0.3, 'wspace': 0.2,
                         'width_ratios': [0.70, 0.30]}
        )
        ax = axes[0]
        for data, label, color in zip(
            [self.train, self.orig],
            ['Train', 'Orig'],
            [self.COLORS['Train'], self.COLORS['Orig']]
        ):
            sns.kdeplot(data=data[self.target], color=color,
                        ax=ax, linewidth=2, label=label)
        ax.set(xlabel='', ylabel='')
        ax.set_title(f'{self.target}  —  KDE')
        ax.legend()
        ax.grid(alpha=0.4)

        ax = axes[1]
        plot_df = pd.concat([
            self.train[[self.target]].assign(Source='Train'),
            self.orig[[self.target]].assign(Source='Orig'),
        ])
        sns.boxplot(data=plot_df, y=self.target, x='Source',
                    palette={'Train': self.COLORS['Train'],
                             'Orig':  self.COLORS['Orig']},
                    width=0.5, linewidth=1, fliersize=1, ax=ax)
        ax.set_title(f'{self.target}  —  Boxplot')
        ax.set(xlabel='', ylabel='')
        ax.grid(alpha=0.4, axis='y')

        plt.tight_layout()
        plt.show()
