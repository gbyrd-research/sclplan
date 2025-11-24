from src.utils.analysis.process_results import aggregate_general_results, aggregate_general_results_compared_to_llmdp, aggregate_consistency_results, general_in_depth_analysis
import os

aggregate_general_results(os.path.join("experiments", "alfworld", "general_low_temp"))
# general_in_depth_analysis(os.path.join("experiments", "alfworld", "general_low_temp"))
# aggregate_general_results_compared_to_llmdp(os.path.join("experiments", "alfworld", "general"))
# experiment_folder = os.path.join("experiments", "alfworld", "consistency_low_temp")
# aggregate_consistency_results(experiment_folder)