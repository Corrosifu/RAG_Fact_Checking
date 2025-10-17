import matplotlib.pyplot as plt

class Visualizer:
    def plot_scores(self, df):
        numeric_df = df.select_dtypes(include="number")

        if "mean_score" in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=["mean_score"])

        mean_scores = numeric_df.mean()
        mean_scores.plot(kind="bar", title="Average RAGAS Scores")
        plt.xticks(rotation=45)
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()


    def preview(self, df):
        print("📊 Preview of results:")
        print(df.head())
