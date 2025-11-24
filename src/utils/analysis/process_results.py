import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from pathlib import Path
from collections import Counter


def get_subdirectories(path):
    return [d.name for d in Path(path).iterdir() if d.is_dir()]


def process_results(experiment_results_path: str):
    # get the folder paths
    task_folders = [
        d.name
        for d in Path(experiment_results_path).iterdir()
        if d.is_dir() and ".hydra" not in d.name
    ]
    num_tasks = len(task_folders)

    analysis = dict()

    analysis["Token Count for Successful Tasks"] = 0
    analysis["Env Steps for Successful Tasks"] = 0

    for result_folder in task_folders:
        results_file_path = os.path.join(
            experiment_results_path, result_folder, "results.json"
        )
        with open(results_file_path, "r") as file:
            results = json.load(file)

            for k, v in results.items():
                if k not in analysis:
                    if isinstance(v, bool):
                        analysis[k] = int(v)
                    else:
                        analysis[k] = v
                else:
                    if not isinstance(v, str):
                        if isinstance(v, bool):
                            analysis[k] += int(v)
                        else:
                            analysis[k] += v
        
        # add the results only for the successful tasks
        if results["Task Success"]:
            analysis["Token Count for Successful Tasks"] += results["Token Count"]
            analysis["Env Steps for Successful Tasks"] += results["Env Steps"]
                     
    avg_analysis = dict()
    for k, v in analysis.items():
        if not isinstance(v, str):
            avg_analysis[k] = v / num_tasks
        else:
            avg_analysis[k] = v

    # overwrite the below with the accurate data
    if analysis["Task Success"] == 0:
        avg_analysis["Token Count for Successful Tasks"] = 0
        avg_analysis["Env Steps for Successful Tasks"] = 0
    else:
        avg_analysis["Token Count for Successful Tasks"] = analysis["Token Count for Successful Tasks"] / analysis["Task Success"]
        avg_analysis["Env Steps for Successful Tasks"] = analysis["Env Steps for Successful Tasks"] / analysis["Task Success"]

    analysis_save_path = os.path.join(experiment_results_path, "analysis.json")
    avg_analysis_save_path = os.path.join(experiment_results_path, "avg_analysis.json")
    with open(analysis_save_path, "w") as file:
        json.dump(analysis, file, indent=3)
    with open(avg_analysis_save_path, "w") as file:
        json.dump(avg_analysis, file, indent=3)

def general_in_depth_analysis(experiment_path: str) -> None:
    # get the results for each llm and method in the experiments folder
    llm_dirs = glob.glob(os.path.join(experiment_path, "*"))

    aggregated_results = dict()

    for llm_dir in llm_dirs:
        if not os.path.isdir(llm_dir):
            continue
        llm = os.path.basename(llm_dir)
        aggregated_results[llm] = dict()
        method_dirs = glob.glob(os.path.join(llm_dir, "*"))
        for method_dir in method_dirs:
            method = os.path.basename(method_dir)
            avg_results_dir = os.path.join(method_dir, "avg_analysis.json")
            if not os.path.exists(avg_results_dir):
                continue
            with open(avg_results_dir, "r") as file:
                aggregated_results[llm][method] = json.load(file)

    # split the results by desired metrics
    metrics = [
        "Task Success",
        "ReAct Proposed w Unsatisfied Preconditions",
        "Precondition Plan Enacted Successfully",
        "Global Planner Plan Enacted Successfully",
        "Num Classical Proposed Actions Successful",
        "Num Precondition Proposed Actions Successful",
        "Num ReAct Proposed Actions Successful",
        "Num Classical Proposed Actions",
        "Num Precondition Proposed Actions",
        "Num ReAct Proposed Actions",
        "Action Sequence Length"
    ]

    llm_names = list()

    metric_results = dict()
    for metric in metrics:
        metric_results[metric] = dict()
        for llm, method_results in aggregated_results.items():
            if "gpt" not in llm:
                continue
            llm_names.append(llm)
            metric_results[metric][llm] = dict()
            for method, results in method_results.items():
                metric_results[metric][llm][method] = results[metric]
    llm_names = list(set(llm_names))

    latex_code_start = r"""\begin{table*}[h]
\centering
\begin{tabular}{cccc|ccc}
\textbf{Metric} & \multicolumn{3}{c}{\textbf{GPT-4o-mini}} & \multicolumn{3}{c}{\textbf{GPT-4o}} \\
& $\theta_{L}$ & $\theta_{L} + PV$ & $\theta_{L} + PV + \theta_{DI}$ & $\theta_{L}$ & $\theta_{L} + PV$ & $\theta_{L} + PV + \theta_{DI}$ \\ \hline \hline \rule{0pt}{2ex}
"""
    lines = [
        r"Task Success (\textuparrow) &" + "%0.3f & %0.3f & %0.3f & %0.3f & %0.3f & %0.3f" % (metric_results["Task Success"]["gpt_4o_mini"]["react"], metric_results["Task Success"]["gpt_4o_mini"]["cala_no_cp"], metric_results["Task Success"]["gpt_4o_mini"]["cala"], metric_results["Task Success"]["gpt_4o"]["react"], metric_results["Task Success"]["gpt_4o"]["cala_no_cp"], metric_results["Task Success"]["gpt_4o"]["cala"]) + r"\\",
        r"$\theta_{L}$ Actions Failing $PV$ (\textdownarrow)" + " & - & %0.3f & %0.3f & - & %0.3f & %0.3f " % (metric_results["ReAct Proposed w Unsatisfied Preconditions"]["gpt_4o_mini"]["cala_no_cp"] / metric_results["Num ReAct Proposed Actions"]["gpt_4o_mini"]["cala_no_cp"], metric_results["ReAct Proposed w Unsatisfied Preconditions"]["gpt_4o_mini"]["cala"] / metric_results["Num ReAct Proposed Actions"]["gpt_4o_mini"]["cala"], metric_results["ReAct Proposed w Unsatisfied Preconditions"]["gpt_4o"]["cala_no_cp"] / metric_results["Num ReAct Proposed Actions"]["gpt_4o"]["cala_no_cp"], metric_results["ReAct Proposed w Unsatisfied Preconditions"]["gpt_4o"]["cala"] / metric_results["Num ReAct Proposed Actions"]["gpt_4o"]["cala"]) + r"\\",
        r"$\theta_{L}$ Failing Actions Corrected w/ $PV$ (\textuparrow)" + " & - & %0.3f & %0.2f & - & %0.3f & %0.3f " % (metric_results["Precondition Plan Enacted Successfully"]["gpt_4o_mini"]["cala_no_cp"] / metric_results["ReAct Proposed w Unsatisfied Preconditions"]["gpt_4o_mini"]["cala_no_cp"], metric_results["Precondition Plan Enacted Successfully"]["gpt_4o_mini"]["cala"] / metric_results["ReAct Proposed w Unsatisfied Preconditions"]["gpt_4o_mini"]["cala"], metric_results["Precondition Plan Enacted Successfully"]["gpt_4o"]["cala_no_cp"] / metric_results["ReAct Proposed w Unsatisfied Preconditions"]["gpt_4o"]["cala_no_cp"], metric_results["Precondition Plan Enacted Successfully"]["gpt_4o"]["cala"] / metric_results["ReAct Proposed w Unsatisfied Preconditions"]["gpt_4o"]["cala"]) +  r"\\",
        r"$\theta_{DI}$ Generates Successful Plan (\textuparrow)" + " & - & - & %0.3f & - & - & %0.3f " % (metric_results["Global Planner Plan Enacted Successfully"]["gpt_4o_mini"]["cala"], metric_results["Global Planner Plan Enacted Successfully"]["gpt_4o"]["cala"]) + r"\\",
        r"Total $\theta_{L}$ Actions" + " & %0.3f & %0.3f & %0.3f & %0.3f & %0.3f & %0.3f " % (metric_results["Num ReAct Proposed Actions"]["gpt_4o_mini"]["react"] / metric_results["Action Sequence Length"]["gpt_4o_mini"]["react"], metric_results["Num ReAct Proposed Actions"]["gpt_4o_mini"]["cala_no_cp"] / metric_results["Action Sequence Length"]["gpt_4o_mini"]["cala_no_cp"], metric_results["Num ReAct Proposed Actions"]["gpt_4o_mini"]["cala"] / metric_results["Action Sequence Length"]["gpt_4o_mini"]["cala"], metric_results["Num ReAct Proposed Actions"]["gpt_4o"]["react"] / metric_results["Action Sequence Length"]["gpt_4o"]["react"], metric_results["Num ReAct Proposed Actions"]["gpt_4o"]["cala_no_cp"] / metric_results["Action Sequence Length"]["gpt_4o"]["cala_no_cp"], metric_results["Num ReAct Proposed Actions"]["gpt_4o"]["cala"] / metric_results["Action Sequence Length"]["gpt_4o"]["cala"]) + r"\\", 
        r"Total $PV$ Actions" + " & - & %0.3f & %0.3f & - & %0.3f & %0.3f " % (metric_results["Num Precondition Proposed Actions"]["gpt_4o_mini"]["cala_no_cp"] / metric_results["Action Sequence Length"]["gpt_4o_mini"]["cala_no_cp"], metric_results["Num Precondition Proposed Actions"]["gpt_4o_mini"]["cala"] / metric_results["Action Sequence Length"]["gpt_4o_mini"]["cala"], metric_results["Num Precondition Proposed Actions"]["gpt_4o"]["cala_no_cp"] / metric_results["Action Sequence Length"]["gpt_4o"]["cala_no_cp"], metric_results["Num Precondition Proposed Actions"]["gpt_4o"]["cala"] / metric_results["Action Sequence Length"]["gpt_4o"]["cala"]) + r"\\",
        r"Total $\theta_{DI}$ Actions" + " & - & - & %0.3f & - & - & %0.3f " % (metric_results["Num Classical Proposed Actions"]["gpt_4o_mini"]["cala"] / metric_results["Action Sequence Length"]["gpt_4o_mini"]["cala"], metric_results["Num Classical Proposed Actions"]["gpt_4o"]["cala"] / metric_results["Action Sequence Length"]["gpt_4o"]["cala"]) + r"\\ \rule{0pt}{2ex}",
    ]
    latex_code_end = r"""\end{tabular}
\caption{Performance comparison of different methods across GPT-4o-mini and GPT-4o. Bold values indicate the best performance in each row.}
\label{tab:ai2thor_experiments}
\end{table*}"""

    with open(os.path.join(experiment_path, f"{os.path.basename(experiment_path)}_in_depth_analysis.txt"), "w") as file:
        file.write(latex_code_start + "\n".join(lines) + latex_code_end)

def aggregate_general_results(experiment_path: str) -> None:
    # get the results for each llm and method in the experiments folder
    llm_dirs = glob.glob(os.path.join(experiment_path, "*"))

    aggregated_results = dict()

    for llm_dir in llm_dirs:
        if not os.path.isdir(llm_dir):
            continue
        llm = os.path.basename(llm_dir)
        aggregated_results[llm] = dict()
        method_dirs = glob.glob(os.path.join(llm_dir, "*"))
        for method_dir in method_dirs:
            method = os.path.basename(method_dir)
            avg_results_dir = os.path.join(method_dir, "avg_analysis.json")
            if not os.path.exists(avg_results_dir):
                continue
            with open(avg_results_dir, "r") as file:
                aggregated_results[llm][method] = json.load(file)

    # split the results by desired metrics
    metrics = [
        "Task Success",
        "Token Count for Successful Tasks",
        "Env Steps for Successful Tasks",
        # "Token Count",
        # "Env Steps"
    ]

    llm_names = list()

    metric_results = dict()
    for metric in metrics:
        metric_results[metric] = dict()
        for llm, method_results in aggregated_results.items():
            llm_names.append(llm)
            metric_results[metric][llm] = dict()
            for method, results in method_results.items():
                metric_results[metric][llm][method] = results[metric]
    llm_names = list(set(llm_names))

    # map the names in the results to the names you want to display
    metric_name_mapping = {
        "Task Success": "Task Success (\u2191)",
        "Token Count for Successful Tasks": "Token Count (Successful Tasks) (\u2193)",
        "Env Steps for Successful Tasks": "Env Steps (Successful Tasks) (\u2193)",
        # "Token Count": "Token Count (\u2193)",
        # "Env Steps": "Env Steps (\u2193)"
    }

    llm_name_mapping = {
        "llama3b": "llama 3b",
        "llama8b": "llama 8b",
        "llama70b": "llama 70b",
        "gpt-3_5-turbo": "gpt-3.5-turbo",
        "gpt_4o_mini": "gpt-4o-mini",
        "gpt_4o": "gpt-4o"
    }

    llm_names = list(llm_name_mapping.keys())

    # method_name_mapping = {
    #     "react": r'$\theta_{L}$',
    #     "cala_no_cp": r'$\theta_{L} + PV$',
    #     "cala": r'$\theta_{L} + PV + \theta_{DI}$',
    #     # "llmdp": "LLM-DP"
    # }

    method_name_mapping = {
        "react": r'$ReAct$',
        "cala_no_cp": r'$ReAct + PV$',
        "cala": r'$SCLPlan$',
        # "llmdp": "LLM-DP"
    }

    # define figure layout
    num_metrics = len(metric_results)
    # fig, axes = plt.subplots((2, 3), num_metrics, figsize=(6 * num_metrics, 5), sharey=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    axes = axes.flatten()

    if num_metrics == 1:
        axes = [axes]  # Ensure axes is iterable for a single metric

    colors = ["purple", "red", "orange", "green", "blue", "cyan"]

    # Define colors for each method
    colors = {llm_names[i]: colors[i] for i in range(len(llm_names))}

    # explicitely set the order of everything
    metric_order = ["Task Success", "Token Count for Successful Tasks", "Env Steps for Successful Tasks"] #, "Token Count", "Env Steps"]
    llm_order = ["llama3b", "llama8b", "llama70b", "gpt-3_5-turbo", "gpt_4o_mini", "gpt_4o"]
    method_order = ["react", "cala_no_cp", "cala"]#, "llmdp"]
    ordered_metric_results = dict()
    for metric in metric_order:
        ordered_metric_results[metric] = dict()
        for llm in llm_order:
            ordered_metric_results[metric][llm] = dict()
            for method in method_order:
                ordered_metric_results[metric][llm][method] = metric_results[metric][llm][method]


    # Loop through each metric to create a subplot
    for ax, (metric, metric_data) in zip(axes, ordered_metric_results.items()):
        df = pd.DataFrame(metric_data).fillna(0)  # Convert to DataFrame, fill missing values with 0

        num_methods = len(df.index)
        num_llms = len(df.columns)
        bar_width = 0.10
        x = np.arange(num_methods)  # X positions for methods

        # Plot bars for each LLM (now coloring by LLM instead of method)
        for i, llm in enumerate(df.columns):
            ax.bar(x + i * bar_width, df.loc[:, llm], width=bar_width, label=llm_name_mapping[llm], color=colors[llm])

        # Set x-axis labels (Methods)
        ax.set_xticks(x + (num_llms / 2 - 0.5) * bar_width)  # Center labels
        ax.set_xticklabels([method_name_mapping[x] for x in df.index], rotation=0)

        # Labels and title
        ax.set_title(metric_name_mapping[metric])
        ax.legend(title="LLM")

    # Adjust layout and save the figure
    plt.tight_layout()
    # save_path = os.path.join(experiment_path, f"{os.path.basename(experiment_path)}_results.png")
    save_path = "temp.png"
    plt.savefig(save_path, dpi=300)
    plt.show()


    percentage_improvement = dict()
    for metric_name, metric_results in ordered_metric_results.items():
        percentage_improvement[metric_name] = dict()
        for llm_name, llm_results in metric_results.items():
            improvement_pv_absolute = llm_results["cala_no_cp"] - llm_results["react"]
            improvement_pv_percentage = (improvement_pv_absolute / llm_results["react"]) * 100
            improvement_cp_absolute = llm_results["cala"] - llm_results["react"]
            improvement_cp_percentage = (improvement_cp_absolute / llm_results["react"]) * 100
            percentage_improvement[metric_name][llm_name] = {
                "pv_absolute": improvement_pv_absolute,
                "pv_percentage": improvement_pv_percentage,
                "cp_absolute": improvement_cp_absolute,
                "cp_percentage": improvement_cp_percentage
            }
    for metric_name, metric_results in percentage_improvement.items():
        percentage_improvement[metric_name]["total"] = {
            "pv_absolute": sum([x["pv_absolute"] for _, x in percentage_improvement[metric_name].items()]) / len([x["pv_absolute"] for _, x in percentage_improvement[metric_name].items()]),
            "pv_percentage": sum([x["pv_percentage"] for _, x in percentage_improvement[metric_name].items()]) / len([x["pv_percentage"] for _, x in percentage_improvement[metric_name].items()]),
            "cp_absolute": sum([x["cp_absolute"] for _, x in percentage_improvement[metric_name].items()]) / len([x["cp_absolute"] for _, x in percentage_improvement[metric_name].items()]),
            "cp_percentage": sum([x["cp_percentage"] for _, x in percentage_improvement[metric_name].items()]) / len([x["cp_percentage"] for _, x in percentage_improvement[metric_name].items()])
        }

    with open(os.path.join(experiment_path, f"{os.path.basename(experiment_path)}_ablation_numerical_results.json"), "w") as file:
        json.dump(percentage_improvement, file, indent=3)   

    latex_code_start = r"""\begin{table}
\setlength{\tabcolsep}{2pt}
\centering
\begin{tabular}{lccc} \hline
\rule{0pt}{2.2ex}Method & Task Success (\%) & Token Count & Env Steps \\ \hline\hline \rule{0pt}{2ex}
"""
    lines = [
        r"$\theta_{L}$" +  "& 0 (0) & 0 (0) & 0 (0)" + r"\\",
        r"+ PV$" +  "& %0.1fk (%0.1fk) & %0.1fkk (%0.1fk) & %0.1fk (%0.1fk)" % (percentage_improvement["Task Success"]["total"]["pv_absolute"], percentage_improvement["Task Success"]["total"]["pv_percentage"], percentage_improvement["Token Count for Successful Tasks"]["total"]["pv_absolute"] / 1000, percentage_improvement["Token Count for Successful Tasks"]["total"]["pv_percentage"], percentage_improvement["Env Steps for Successful Tasks"]["total"]["pv_absolute"], percentage_improvement["Env Steps for Successful Tasks"]["total"]["pv_percentage"]) + r"\\",
        r"+ PV + \theta_{formal}$" +  "& %0.1fk (%0.1fk) & %0.1fkk (%0.1fk) & %0.1fk (%0.1fk)" % (percentage_improvement["Task Success"]["total"]["cp_absolute"], percentage_improvement["Task Success"]["total"]["cp_percentage"], percentage_improvement["Token Count for Successful Tasks"]["total"]["cp_absolute"] / 1000, percentage_improvement["Token Count for Successful Tasks"]["total"]["cp_percentage"], percentage_improvement["Env Steps for Successful Tasks"]["total"]["cp_absolute"], percentage_improvement["Env Steps for Successful Tasks"]["total"]["cp_percentage"]) + r"\\"
    ]
    latex_code_end = r"""\hline \rule{0pt}{2ex}
\end{tabular}
\caption{Ablation study improvements.}
\label{tab:baselines}
\end{table}
"""

    with open(os.path.join(experiment_path, f"{os.path.basename(experiment_path)}_ablation_numerical_results_latex.txt"), "w") as file:
        file.write(latex_code_start + "\n".join(lines) + latex_code_end)

def aggregate_general_results_compared_to_llmdp(experiment_path: str) -> None:
    # get the results for each llm and method in the experiments folder
    llm_dirs = glob.glob(os.path.join(experiment_path, "*"))

    aggregated_results = dict()

    for llm_dir in llm_dirs:
        if not os.path.isdir(llm_dir):
            continue
        llm = os.path.basename(llm_dir)
        aggregated_results[llm] = dict()
        method_dirs = glob.glob(os.path.join(llm_dir, "*"))
        for method_dir in method_dirs:
            method = os.path.basename(method_dir)
            avg_results_dir = os.path.join(method_dir, "avg_analysis.json")
            if not os.path.exists(avg_results_dir):
                continue
            with open(avg_results_dir, "r") as file:
                aggregated_results[llm][method] = json.load(file)

    # split the results by desired metrics
    metrics = [
        "Task Success",
        "Token Count for Successful Tasks",
        "Env Steps for Successful Tasks",
        # "Token Count",
        # "Env Steps"
    ]

    llm_names = list()

    metric_results = dict()
    for metric in metrics:
        metric_results[metric] = dict()
        for llm, method_results in aggregated_results.items():
            llm_names.append(llm)
            metric_results[metric][llm] = dict()
            for method, results in method_results.items():
                metric_results[metric][llm][method] = results[metric]
    llm_names = list(set(llm_names))

    # map the names in the results to the names you want to display
    metric_name_mapping = {
        "Task Success": "Task Success (\u2191)",
        "Token Count for Successful Tasks": "Token Count (Successful Tasks) (\u2193)",
        "Env Steps for Successful Tasks": "Env Steps (Successful Tasks) (\u2193)",
        # "Token Count": "Token Count (\u2193)",
        # "Env Steps": "Env Steps (\u2193)"
    }

    llm_name_mapping = {
        "llama3b": "llama 3b",
        "llama8b": "llama 8b",
        "llama70b": "llama 70b",
        "gpt-3_5-turbo": "gpt-3.5-turbo",
        "gpt_4o_mini": "gpt-4o-mini",
        "gpt_4o": "gpt-4o"
    }

    llm_names = list(llm_name_mapping.keys())

    method_name_mapping = {
        # "react": "ZS ReAct",
        # "cala_no_cp": "ZS ReAct + PV",
        "cala": "FALA (Ours)",
        "llmdp": "LLM-DP"
    }

    # define figure layout
    num_metrics = len(metric_results)
    # fig, axes = plt.subplots((2, 3), num_metrics, figsize=(6 * num_metrics, 5), sharey=False)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.3), sharey=False)
    axes = axes.flatten()

    if num_metrics == 1:
        axes = [axes]  # Ensure axes is iterable for a single metric

    colors = ["purple", "red", "orange", "green", "blue", "cyan"]

    # Define colors for each method
    colors = {llm_names[i]: colors[i] for i in range(len(llm_names))}

    # explicitely set the order of everything
    metric_order = ["Task Success", "Token Count for Successful Tasks", "Env Steps for Successful Tasks"] #, "Token Count", "Env Steps"]
    llm_order = ["llama3b", "llama8b", "llama70b", "gpt-3_5-turbo", "gpt_4o_mini", "gpt_4o"]
    method_order = ["cala", "llmdp"]
    ordered_metric_results = dict()
    for metric in metric_order:
        ordered_metric_results[metric] = dict()
        for llm in llm_order:
            ordered_metric_results[metric][llm] = dict()
            for method in method_order:
                ordered_metric_results[metric][llm][method] = metric_results[metric][llm][method]


    # Loop through each metric to create a subplot
    for ax, (metric, metric_data) in zip(axes, ordered_metric_results.items()):
        df = pd.DataFrame(metric_data).fillna(0)  # Convert to DataFrame, fill missing values with 0

        num_methods = len(df.index)
        num_llms = len(df.columns)
        bar_width = 0.08
        x = np.arange(num_methods)  # X positions for methods

        # Plot bars for each LLM (now coloring by LLM instead of method)
        for i, llm in enumerate(df.columns):
            ax.bar(x + i * bar_width, df.loc[:, llm], width=bar_width, label=llm_name_mapping[llm], color=colors[llm])

        # Set x-axis labels (Methods)
        ax.set_xticks(x + (num_llms / 2 - 0.5) * bar_width)  # Center labels
        ax.set_xticklabels([method_name_mapping[x] for x in df.index], rotation=0)

        # Labels and title
        ax.set_title(metric_name_mapping[metric])
        if "Env Steps" in metric:
            ax.legend(title="LLM")

    # Adjust layout and save the figure
    plt.tight_layout()
    save_path = os.path.join(experiment_path, f"{os.path.basename(experiment_path)}_results_compared_to_llmdp.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

def aggregate_consistency_results(experiment_folder=None):

    """This function looks at the consistency across multiple runs on the same task.
    To do so, various metrics are collected for each task. A box plot for 
    """

    # get the results for each llm and method in the experiments folder
    if experiment_folder is None:
        experiment_folder = os.path.join("experiments", "alfworld", "consistency")
    llm_dirs = glob.glob(os.path.join(experiment_folder, "*"))

    results = dict()

    metrics_used_for_consistency = {
        "Task Success",
        "Token Count",
        "Env Steps",
        "Action Sequence Length",
        "Num ReAct Proposed Actions",
        "Num ReAct Proposed Actions Successful"
    }

    for llm_dir in llm_dirs:
        llm = os.path.basename(llm_dir)
        results[llm] = dict()
        task_results = dict() # maps each method task to a list of results from different runs
        method_dirs = glob.glob(os.path.join(llm_dir, "*"))
        for method_dir in method_dirs:
            method = os.path.basename(method_dir)
            results[llm][method] = dict()
            run_dirs = [x for x in glob.glob(os.path.join(method_dir, "*")) if ".hydra" not in x]
            for run_dir in run_dirs:
                run_idx = os.path.basename(run_dir)
                task_dirs = [x for x in glob.glob(os.path.join(run_dir, "*")) if os.path.isdir(x)]
                for task_dir in task_dirs:
                    task_idx = int(os.path.basename(task_dir).split("_")[-1])
                    if task_idx not in results[llm][method]:
                        results[llm][method][task_idx] = dict()
                    # read the task results
                    if not os.path.exists(os.path.join(task_dir, "results.json")):
                        print(f"WARNING: results.json file not found in {task_dir}!!")
                        continue
                    with open(os.path.join(task_dir, "results.json"), "r") as file:
                        task_results = json.load(file)
                    for metric in metrics_used_for_consistency:
                        if metric not in results[llm][method][task_idx]:
                            results[llm][method][task_idx][metric] = list()
                        results[llm][method][task_idx][metric].append(task_results[metric])
    
    # visualization 1: get the average standard deviation across the 10 tasks for each llm, method, and metric
    avg_std_dev = dict()
    avg_mean = dict() # this is the average success percentage on 10 runs of the same task averaged across 10 different tasks
    for llm, llm_results in results.items():
        avg_std_dev[llm] = dict()
        avg_mean[llm] = dict()
        for method, method_results in llm_results.items():
            avg_std_dev[llm][method] = dict()
            avg_mean[llm][method] = dict()
            for run, run_results in method_results.items():
                for metric, metric_results in run_results.items():
                    if metric not in avg_std_dev[llm][method]:
                        avg_std_dev[llm][method][metric] = list()
                    if metric not in avg_mean[llm][method]:
                        avg_mean[llm][method][metric] = list()
                    if len(metric_results) == 1 or len(metric_results) == 0:
                        print("WARNING: Metric results are incomplete.")
                        avg_std_dev[llm][method][metric].append(0)
                        avg_mean[llm][method][metric].append(0)
                        continue
                    avg_std_dev[llm][method][metric].append(statistics.stdev(metric_results))
                    avg_mean[llm][method][metric].append(statistics.mean(metric_results))
            for metric in metrics_used_for_consistency:
                avg_std_dev[llm][method][metric] = statistics.mean(avg_std_dev[llm][method][metric])
                avg_mean[llm][method][metric] = statistics.mean(avg_mean[llm][method][metric])

    save_file = os.path.join(experiment_folder, "Consistency_Benchmark_Average_Standard_Deviation_Across_Task_Runs.json")
    with open(save_file, "w") as file:
        json.dump(avg_std_dev, file, indent=3)

    # create the latex table and save it

    top = r"""\begin{table}
\setlength{\tabcolsep}{2pt}
\centering
\begin{tabular}{clccc} \hline
\rule{0pt}{2.2ex}LLM & Method & Task Success & Token Count & Env Steps \\ \hline\hline \rule{0pt}{2ex}"""

    lines = [
        "\multirow{3}{*}{Llama 3B} & " + r"$\theta_{L}$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["llama3b"]["react"]["Task Success"], avg_std_dev["llama3b"]["react"]["Task Success"], avg_mean["llama3b"]["react"]["Token Count"] / 1000, avg_std_dev["llama3b"]["react"]["Token Count"] / 1000, avg_mean["llama3b"]["react"]["Env Steps"], avg_std_dev["llama3b"]["react"]["Env Steps"]) + r"\\",
        "& " + r"+ $PV$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["llama3b"]["cala_no_cp"]["Task Success"], avg_std_dev["llama3b"]["cala_no_cp"]["Task Success"], avg_mean["llama3b"]["cala_no_cp"]["Token Count"] / 1000, avg_std_dev["llama3b"]["cala_no_cp"]["Token Count"] / 1000, avg_mean["llama3b"]["cala_no_cp"]["Env Steps"], avg_std_dev["llama3b"]["cala_no_cp"]["Env Steps"]) + r"\\",
        "& " + r"+ $PV + \theta_{DI}$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["llama3b"]["cala"]["Task Success"], avg_std_dev["llama3b"]["cala"]["Task Success"], avg_mean["llama3b"]["cala"]["Token Count"] / 1000, avg_std_dev["llama3b"]["cala"]["Token Count"] / 1000, avg_mean["llama3b"]["cala"]["Env Steps"], avg_std_dev["llama3b"]["cala"]["Env Steps"]) + r"\\",
        r"\hline \rule{0pt}{2ex}",
        "\multirow{3}{*}{Llama 8B} & " + r"$\theta_{L}$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["llama8b"]["react"]["Task Success"], avg_std_dev["llama8b"]["react"]["Task Success"], avg_mean["llama8b"]["react"]["Token Count"] / 1000, avg_std_dev["llama8b"]["react"]["Token Count"] / 1000, avg_mean["llama8b"]["react"]["Env Steps"], avg_std_dev["llama8b"]["react"]["Env Steps"]) + r"\\",
        "& " + r"+ $PV$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["llama8b"]["cala_no_cp"]["Task Success"], avg_std_dev["llama8b"]["cala_no_cp"]["Task Success"], avg_mean["llama8b"]["cala_no_cp"]["Token Count"] / 1000, avg_std_dev["llama8b"]["cala_no_cp"]["Token Count"] / 1000, avg_mean["llama8b"]["cala_no_cp"]["Env Steps"], avg_std_dev["llama8b"]["cala_no_cp"]["Env Steps"]) + r"\\",
        "& " + r"+ $PV + \theta_{DI}$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["llama8b"]["cala"]["Task Success"], avg_std_dev["llama8b"]["cala"]["Task Success"], avg_mean["llama8b"]["cala"]["Token Count"] / 1000, avg_std_dev["llama8b"]["cala"]["Token Count"] / 1000, avg_mean["llama8b"]["cala"]["Env Steps"], avg_std_dev["llama8b"]["cala"]["Env Steps"]) + r"\\",
        r"\hline \rule{0pt}{2ex}",
        "\multirow{3}{*}{Llama 70B} & " + r"$\theta_{L}$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["llama70b"]["react"]["Task Success"], avg_std_dev["llama70b"]["react"]["Task Success"], avg_mean["llama70b"]["react"]["Token Count"] / 1000, avg_std_dev["llama70b"]["react"]["Token Count"] / 1000, avg_mean["llama70b"]["react"]["Env Steps"], avg_std_dev["llama70b"]["react"]["Env Steps"]) + r"\\",
        "& " + r"+ $PV$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["llama70b"]["cala_no_cp"]["Task Success"], avg_std_dev["llama70b"]["cala_no_cp"]["Task Success"], avg_mean["llama70b"]["cala_no_cp"]["Token Count"] / 1000, avg_std_dev["llama70b"]["cala_no_cp"]["Token Count"] / 1000, avg_mean["llama70b"]["cala_no_cp"]["Env Steps"], avg_std_dev["llama70b"]["cala_no_cp"]["Env Steps"]) + r"\\",
        "& " + r"+ $PV + \theta_{DI}$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["llama70b"]["cala"]["Task Success"], avg_std_dev["llama70b"]["cala"]["Task Success"], avg_mean["llama70b"]["cala"]["Token Count"] / 1000, avg_std_dev["llama70b"]["cala"]["Token Count"] / 1000, avg_mean["llama70b"]["cala"]["Env Steps"], avg_std_dev["llama70b"]["cala"]["Env Steps"]) + r"\\",
        r"\hline \rule{0pt}{2ex}",
        "\multirow{3}{*}{GPT-4o-mini} & " + r"$\theta_{L}$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)"  % (avg_mean["gpt_4o_mini"]["react"]["Task Success"], avg_std_dev["gpt_4o_mini"]["react"]["Task Success"], avg_mean["gpt_4o_mini"]["react"]["Token Count"] / 1000, avg_std_dev["gpt_4o_mini"]["react"]["Token Count"] / 1000, avg_mean["gpt_4o_mini"]["react"]["Env Steps"], avg_std_dev["gpt_4o_mini"]["react"]["Env Steps"]) + r"\\",
        "& " + r"+ $PV$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["gpt_4o_mini"]["cala_no_cp"]["Task Success"], avg_std_dev["gpt_4o_mini"]["cala_no_cp"]["Task Success"], avg_mean["gpt_4o_mini"]["cala_no_cp"]["Token Count"] / 1000, avg_std_dev["gpt_4o_mini"]["cala_no_cp"]["Token Count"] / 1000, avg_mean["gpt_4o_mini"]["cala_no_cp"]["Env Steps"], avg_std_dev["gpt_4o_mini"]["cala_no_cp"]["Env Steps"]) + r"\\",
        "& " + r"+ $PV + \theta_{DI}$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["gpt_4o_mini"]["cala"]["Task Success"], avg_std_dev["gpt_4o_mini"]["cala"]["Task Success"], avg_mean["gpt_4o_mini"]["cala"]["Token Count"] / 1000, avg_std_dev["gpt_4o_mini"]["cala"]["Token Count"] / 1000, avg_mean["gpt_4o_mini"]["cala"]["Env Steps"], avg_std_dev["gpt_4o_mini"]["cala"]["Env Steps"]) + r"\\",
        r"\hline \rule{0pt}{2ex}",
        "\multirow{3}{*}{GPT-4o} & " + r"$\theta_{L}$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["gpt_4o"]["react"]["Task Success"], avg_std_dev["gpt_4o"]["react"]["Task Success"], avg_mean["gpt_4o"]["react"]["Token Count"] / 1000, avg_std_dev["gpt_4o"]["react"]["Token Count"] / 1000, avg_mean["gpt_4o"]["react"]["Env Steps"], avg_std_dev["gpt_4o"]["react"]["Env Steps"]) + r"\\",
        "& " + r"+ $PV$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["gpt_4o"]["cala_no_cp"]["Task Success"], avg_std_dev["gpt_4o"]["cala_no_cp"]["Task Success"], avg_mean["gpt_4o"]["cala_no_cp"]["Token Count"] / 1000, avg_std_dev["gpt_4o"]["cala_no_cp"]["Token Count"] / 1000, avg_mean["gpt_4o"]["cala_no_cp"]["Env Steps"], avg_std_dev["gpt_4o"]["cala_no_cp"]["Env Steps"]) + r"\\",
        "& " + r"+ $PV + \theta_{DI}$" + " & %0.2f (\u00B1%0.2f) & %0.1fk (\u00B1%0.1fk) & %0.2f (\u00B1%0.2f)" % (avg_mean["gpt_4o"]["cala"]["Task Success"], avg_std_dev["gpt_4o"]["cala"]["Task Success"], avg_mean["gpt_4o"]["cala"]["Token Count"] / 1000, avg_std_dev["gpt_4o"]["cala"]["Token Count"] / 1000, avg_mean["gpt_4o"]["cala"]["Env Steps"], avg_std_dev["gpt_4o"]["cala"]["Env Steps"]) + r"\\",
    ]    

    bottom = r"""\hline \rule{0pt}{2ex}
\end{tabular}
\caption{Ablation study using consistency metrics. The values presented are the average standard deviation of the metrics across 10 subsequent runs of the same 10 tasks.}
\label{tab:ablation_consistency}
\end{table}"""

    lines = "\n".join(lines)

    latex_code = top + lines + bottom

    save_file = os.path.join(experiment_folder, "Consistency_Benchmark_Average_Standard_Deviation_Across_Task_Runs_Latex.txt")

    with open(save_file, "w") as file:
        file.write(latex_code)

if __name__=="__main__":
    # process_results("")
    general_in_depth_analysis("/home/local_byrd/Desktop/concept-agent/experiments/alfworld/general_low_temp")