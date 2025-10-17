
from Evaluation.dataset_builder import DatasetBuilder
from Evaluation.metric_runner import MetricRunner
from Evaluation.logger import EvalLogger
from Evaluation.visualization import Visualizer
from config import DEFAULT_METRICS

class Eval_Pipeline:

    def __init__(self, retriever, generator, metrics=DEFAULT_METRICS):
        self.dataset_builder = DatasetBuilder(retriever, generator)
        self.metric_runner = MetricRunner(metrics)
        self.logger = EvalLogger()
        self.visualizer = Visualizer()

    def run(self, test_set):
        dataset = self.dataset_builder.build(test_set)
        print(dataset)
        df = self.metric_runner.run(dataset)
        self.logger.save_csv(df)
        self.logger.log_to_langsmith(df)
        self.visualizer.preview(df)
        self.visualizer.plot_scores(df)
        return df


