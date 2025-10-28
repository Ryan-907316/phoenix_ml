# report_generation.py
# This module takes all the information gathered during the workflow and generates a .pdf report with all the information.
# I decided to add this as there was a lot of information being processed and it's nice for reproducibility and organisation.
# Please note that reports can reach multiple dozens of pages long and contain a lot of images. Some of these images can be hard to read, I'm working on it.
# For more information refer to the reportlab documentation.

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from datetime import datetime
from reportlab.platypus import Image
import matplotlib.pyplot as plt

import os

from phoenix_ml.system_info import SystemInfo
from phoenix_ml.data_preprocessing import *
from phoenix_ml.uncertainty_quantification import save_uncertainty_results

# Initialisation of the .pdf document
def init_pdf_report(
    filename="system_report.pdf",
    output_dir=".",
    title="Machine Learning Report",
    font_name="Helvetica",
    font_size=10,
    title_font_size=18,
    heading_font_size=14
):
    # Initialises the .pdf document and returns the doc object, elements list, and styles.
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    doc = SimpleDocTemplate(
        filepath, pagesize=A4, rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20
    )
    elements = []

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CustomTitle", fontName=font_name, fontSize=title_font_size,
                              leading=title_font_size+4, alignment=1))
    styles.add(ParagraphStyle(name="CustomHeading", fontName=font_name, fontSize=heading_font_size,
                              spaceAfter=10, leading=heading_font_size+2))
    styles.add(ParagraphStyle(name="CustomBody", fontName=font_name, fontSize=font_size,
                              spaceAfter=6, leading=font_size+2))
    styles.add(ParagraphStyle(
    name="CustomSubheading",
    parent=styles["Heading2"],  # You can use Heading3 if you want it smaller
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=14,
    alignment=TA_LEFT,
    spaceAfter=6
))
    # Add title and timestamp
    elements.append(Paragraph(title, styles["CustomTitle"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["CustomBody"]))
    elements.append(Spacer(1, 20))

    return doc, elements, styles, filepath


def add_system_info_to_pdf(elements, styles, font_name="Helvetica", font_size=10):
    # Adds system information to the PDF content (elements list).

    sysinfo = SystemInfo()
    info = sysinfo.gather()

    table_data = [["Feature", "Details"]] + list(zip(info["Feature"], info["Details"]))
    table = Table(table_data, hAlign='LEFT', colWidths=[60*mm, 100*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), font_size),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))

    elements.append(Paragraph("System Information:", styles["CustomHeading"]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # GPU Note
    elements.append(Paragraph(
    "Note: GPU acceleration in this workflow is optimised for NVIDIA GPUs using CUDA.<br/> "
    "This is because popular Machine Learning frameworks like PyTorch rely on CUDA, a proprietary technology developed by NVIDIA for GPU acceleration. "
    "While there are alternative frameworks and libraries, such as ROCm for AMD GPUs or oneAPI for Intel GPUs, these are not yet universally supported or integrated in many ML workflows. "
    "As a result, this workflow defaults to CUDA for GPU acceleration.<br/><br/>"
    "For systems with AMD GPUs, users may explore ROCm for compatibility with specific frameworks. "
    "Similarly, Intel GPU users can consider Intel oneAPI. Note that additional setup may be required to enable GPU support with these alternatives. "
    "If no compatible GPU is detected, the workflow will default to using the CPU, which may significantly increase computation time.<br/><br/>"
    "For more details on GPU support, you can explore the following resources:<br/>"
    "CUDA (NVIDIA): https://docs.nvidia.com/cuda/<br/>"
    "ROCm (AMD): https://rocm.docs.amd.com/en/latest/<br/>"
    "oneAPI (Intel): https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html",
    styles["CustomBody"]
))

def save_preprocessing_plots(results, output_dir, prefix="preprocessing"):
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    for label, fig in results["figures"].items():
        path = os.path.join(output_dir, f"{prefix}_{label.lower().replace(' ', '_')}.png")
        fig.savefig(path)
        plot_paths[label] = path
        plt.close(fig)  # Close after saving

    return plot_paths


def add_preprocessing_section(elements, results, plot_paths, dataset_path, styles):
    elements.append(Paragraph("Preprocessing Summary", styles["CustomHeading"]))

    meta = results.get("meta", {})
    # Dataset Overview table
    wrap = ParagraphStyle(name="WrapSmall", fontSize=8, leading=9, wordWrap="CJK")
    table_data = [
        ["Dataset path", Paragraph(str(meta.get("dataset_path", dataset_path)), wrap)],
        ["Rows × Columns", f"{meta.get('n_rows','?')} × {meta.get('n_cols','?')}"],
        ["Targets", Paragraph(", ".join(meta.get("targets", [])) or "—", wrap)],
        ["# Features", str(meta.get("n_features", "?"))],
        ["Train/Test split",
         f"Train: {meta.get('train_count','?')} ({meta.get('train_prop',0):.1%})  |  "
         f"Test: {meta.get('test_count','?')} ({meta.get('test_prop',0):.1%})  "
         f"[method: {meta.get('split_method','?')}, requested test_size={meta.get('test_size_param','?')}]"],
    ]
    tbl = Table(table_data, colWidths=[40*mm, 120*mm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#eeeeee")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 6))

    # Feature list (wrapped)
    feat_list = meta.get("features", [])
    if feat_list:
        features_par = Paragraph(f"<b>Features ({len(feat_list)}):</b> " + ", ".join(map(str, feat_list)), wrap)
        elements.append(features_par)
        elements.append(Spacer(1, 12))

    # Plot captions
    captions = {
        "Target vs Target": (
            "Below shows a scatter plot between two (or more) targets, "
            "separating training and test points. Black dots indicate the training data and red dots the testing data. "
        ),
        "_features_vs_": (
            "Feature-vs-Target scatter grid for the training data. Each subplot shows a single feature against "
            "the target with a linear line of best fit to indicate the average trend. "
            "A tight, linear band suggests strong linear correlation; funnel shapes or curves suggest "
            "heteroscedasticity or non-linearity."
        ),
        "Boxplots": (
            "Boxplots summarise distributions for features and targets (median, interquartile range, and outliers). "
            "Long whiskers/head-heavy tails indicate skew; many points beyond whiskers indicate potential outliers."
            "See the legend of each plot for more information."
        ),
        "Distance Correlation": (
            "Distance correlation matrix across input features. Distance correlation captures general (including "
            "non-linear) dependence; values near 1 suggest strong dependence, while values near 0 suggest weak or no "
            "dependence. Use this to spot redundant features or feature clusters. "
            "An optional dummy variable can be added, serving as a control variable and values above the dummy indicate more than random dependence."
        ),
    }

    # Render plots with their richer captions
    for label, img_path in plot_paths.items():
        if not os.path.exists(img_path):
            continue

        # Pick caption
        if label.startswith("Features vs "):
            caption = captions["_features_vs_"]
        else:
            caption = captions.get(label, label)

        elements.append(Paragraph(caption, styles["CustomBody"]))
        elements.append(Image(img_path, width=160*mm, height=100*mm))
        elements.append(Spacer(1, 12))

def add_model_selection_section(elements, styles, selected_model_names, preferred_model_name=None):
    elements.append(Paragraph("Selected Models", styles["CustomHeading"]))
    elements.append(Paragraph(
        "The following machine learning models were selected by the user for training and evaluation.",
        styles["CustomBody"]
    ))
    elements.append(Spacer(1, 6))

    # Bullet list of models
    for model in selected_model_names:
        if preferred_model_name and model == preferred_model_name:
            model_text = f"<b>• {model} (Preferred for Interpretability)</b>"
        else:
            model_text = f"• {model}"
        elements.append(Paragraph(model_text, styles["CustomBody"]))

    elements.append(Spacer(1, 12))


def add_model_training_table_to_report(elements, results_df, styles, max_rows=100):
    # Adds the model training results DataFrame as a table to the PDF report.

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Model Training Results", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    if results_df.empty:
        elements.append(Paragraph("No model training results to display.", styles["CustomBody"]))
        return

    # Truncate the table if it has too many rows
    if len(results_df) > max_rows:
        display_df = results_df.head(max_rows).copy()
        truncated = True
    else:
        display_df = results_df.copy()
        truncated = False

    # Convert the DataFrame to a list of lists (rows)
    data = [display_df.columns.tolist()] + display_df.round(4).values.tolist()

    # Create the table
    table = Table(data, repeatRows=1, colWidths=[4 * cm] + [2.5 * cm] * (len(data[0]) - 1))

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d0d0d0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ]))

    elements.append(table)

    if truncated:
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(
            f"Note: Only the first {max_rows} rows are shown for brevity. Full results are available in the .csv export.",
            styles["CustomBody"]
        ))

def add_uq_section(elements, uq_plot_paths, styles, stage="Before HPO", uq_settings=None):
    elements.append(Paragraph(f"Uncertainty Quantification ({stage})", styles["CustomHeading"]))

    # Description
    elements.append(Paragraph(
        "This section shows prediction intervals or confidence intervals for each model and target variable "
        f"at the '{stage}' stage. For full metrics, refer to the .csv files for each stage.",
        styles["CustomBody"]
    ))

    # Settings block
    if uq_settings:
        method_used = uq_settings.get("uq_method", "N/A")
        n_bootstrap = uq_settings.get("n_bootstrap", "N/A")
        conf_interval = uq_settings.get("confidence_interval", "N/A")
        calibration_frac = uq_settings.get("calibration_frac", "N/A")
        test_size = uq_settings.get("subsample_test_size", "N/A")

        settings_html = f"""
            <b>UQ Method:</b> {method_used}<br/>
            <b>Number of Bootstraps:</b> {n_bootstrap}<br/>
            <b>Confidence Interval (CI):</b> {conf_interval}%<br/>
            <b>Calibration Fraction:</b> {calibration_frac}<br/>
            <b>Subsample Test Size:</b> {test_size}
        """
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(settings_html, styles["CustomBody"]))

    elements.append(Spacer(1, 12))

    # Add plots
    for label, img_path in uq_plot_paths.items():
        if os.path.exists(img_path):
            elements.append(Paragraph(label, styles["CustomBody"]))
            elements.append(Image(img_path, width=170*mm, height=100*mm))
            elements.append(Spacer(1, 12))
        else:
            elements.append(Paragraph(f"Could not find UQ image: {img_path}", styles["CustomBody"]))
            elements.append(Spacer(1, 6))

def handle_uq_reporting_section(
    uq_df,
    uq_figures,
    stage_label,
    elements,
    styles,
    image_output_dir,
    csv_output_dir,
    uq_settings: dict = None
):
    os.makedirs(image_output_dir, exist_ok=True)
    uq_plot_paths = {}

    for model_name, fig in uq_figures.items():
        clean_model = model_name.replace(' ', '_')
        clean_stage = stage_label.replace(' ', '_').replace("_Before_HPO", "")
        fname = f"UQ_{clean_model}_{clean_stage}.png"
        fpath = os.path.join(image_output_dir, fname)
        fig.savefig(fpath)
        plt.close(fig)
        uq_plot_paths[model_name] = fpath

    add_uq_section(elements, uq_plot_paths, styles, stage=stage_label, uq_settings=uq_settings)
    save_uncertainty_results(uq_df, results_dir=csv_output_dir, stage=stage_label)


def add_interpretability_section(
    elements,
    interpretability_figures,
    styles,
    output_dir,
    settings
):

    # Add ICE, PDP, and SHAP interpretability plots to the PDF.

    os.makedirs(output_dir, exist_ok=True)

    elements.append(Paragraph("Interpretability Analysis", styles["CustomHeading"]))

    # Settings description
    setting_text = (
        f"This section shows interpretability plots generated using the following settings:<br/>"
        f"<b>Preferred model:</b> {settings.get('preferred_model_name')}<br/>"
        f"<b>Test sample size:</b> {settings.get('test_sample_size')} samples<br/>"
        f"<b>Background sample size:</b> {settings.get('background_sample_size')}<br/>"
        f"<b>ICE/PDP subsample:</b> {settings.get('subsample')}<br/>"
        f"<b>Grid resolution:</b> {settings.get('grid_resolution')}<br/>"
    )
    elements.append(Paragraph(setting_text, styles["CustomBody"]))
    elements.append(Spacer(1, 12))

    # Plot type descriptions
    descriptions = {
        "ICE/PDP": "ICE and PDP plots show how individual features affect model predictions. ICE (blue) shows per-sample curves, while PDP (orange) shows the average trend.",
        "SHAP_Summary": "SHAP summary plots show the overall importance and distribution of SHAP values across all features.",
        "SHAP_Dependence": "SHAP dependence plots illustrate how each feature's value relates to its SHAP value, indicating feature interaction and impact."
    }

    for fig_name, fig in interpretability_figures.items():
        fig_path = os.path.join(output_dir, f"{fig_name.replace(' ', '_')}.png")
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

        # Split figure name to extract type and target
        parts = fig_name.split("_")
        if len(parts) >= 3:
            label_type = "_".join(parts[:2])  # e.g., SHAP_Summary
            target_name = "_".join(parts[2:])  # e.g., Motor_Speed
        else:
            label_type = parts[0]
            target_name = "_".join(parts[1:]) if len(parts) > 1 else "Unknown Target"

        # Add figure to PDF
        elements.append(Paragraph(f"{label_type.replace('_', ' ')} for {target_name}", styles["CustomBody"]))
        elements.append(Paragraph(descriptions.get(label_type, "Interpretability plot."), styles["CustomBody"]))
        elements.append(Image(fig_path, width=170*mm, height=100*mm))
        elements.append(Spacer(1, 12))

def add_hpo_summary_section(
    elements,
    styles,
    hpo_metrics: dict,
    hpo_params: dict,
    hpo_times: dict,
    hpo_plots: dict,
    methods_used: list,
    metric_used: str,
    sampling_method: str,
    sample_size: int,
    n_iter: int,
    evals: int,
    calls: int,
    n_jobs: int,
    csv_path: str,
    best_models_per_target,
    output_dir: str = "report_images"
):
    if isinstance(best_models_per_target, dict):
        best_models_per_target = pd.DataFrame.from_dict(best_models_per_target, orient="index")
        
        # Drop the 'Target Variable' column if it already exists
        if 'Target Variable' in best_models_per_target.columns:
            best_models_per_target.drop(columns=['Target Variable'], inplace=True)
            
        # Reset index and rename properly
        best_models_per_target.index.name = "Target Variable"
        best_models_per_target = best_models_per_target.reset_index()

    # Paragraph style for wrapping long param strings
    param_style = ParagraphStyle(
        name="ParamStyle",
        fontSize=9,
        leading=10,
        wordWrap='CJK'
    )
    wrap_style = ParagraphStyle(
        name="WrappedCell",
        fontSize=6.5,
        leading=7.5,
        wordWrap='CJK'
        )
    # Title
    elements.append(Paragraph("Hyperparameter Optimisation (HPO)", styles["CustomHeading"]))

    # Intro and settings
    methods_description = {
        "random": f"Random Search ({sampling_method}, {n_iter} iterations, sample size = {sample_size})",
        "hyperopt": f"Hyperopt (TPE, {evals} evaluations)",
        "skopt": f"Scikit-Optimize (Bayesian optimisation with Gaussian Processes, {calls} calls)"
    }

    settings_text = "This section presents the best performance for each model and target variable across the selected HPO methods. "
    settings_text += f"The chosen evaluation metric is <b>{metric_used}</b>.<br/><br/>"

    settings_text += "<b>HPO Methods and Settings:</b><br/>"
    for method in methods_used:
        if method in methods_description:
            settings_text += f"• <b>{method.capitalize()}</b>: {methods_description[method]}<br/>"

    elements.append(Paragraph(settings_text, styles["CustomBody"]))
    elements.append(Spacer(1, 12))

    # Per-model breakdown
    model_names = list(hpo_metrics[methods_used[0]].keys())  # Assumes consistent keys

    for model_name in model_names:
        elements.append(Paragraph(f"{model_name}", styles["CustomSubheading"]))

        for method in methods_used:
            if model_name not in hpo_metrics[method]:
                continue

            elements.append(Paragraph(f"<b>Method:</b> {method.capitalize()}", styles["CustomBody"]))

            # Table Header
            table_data = [["Target Variable", f"Best {metric_used}", "Elapsed Time (s)", "Best Parameters"]]

            for target_var, metric_data in hpo_metrics[method][model_name].items():
                best_score = metric_data.get(metric_used)
                elapsed = round(metric_data.get("elapsed_time", 0), 2)
                best_params = hpo_params[method][model_name].get(target_var, {})
                params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
                params_paragraph = Paragraph(params_str if params_str else "N/A", param_style)

                table_data.append([
                    target_var,
                    f"{best_score:.4f}" if best_score is not None else "N/A",
                    f"{elapsed:.2f}",
                    params_paragraph
                ])

            # Style and insert table
            table = Table(table_data, colWidths=[40*mm, 30*mm, 30*mm, 70*mm])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (4, 0), (5, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6.5),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.grey)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 6))

            # Insert existing HPO plot image if available
            plot_path = hpo_plots.get(method, {}).get(model_name)
            if plot_path and os.path.exists(plot_path):
                elements.append(Image(plot_path, width=14*cm, height=7*cm))
                elements.append(Spacer(1, 12))
            else:
                elements.append(Paragraph(f"No HPO plot found for {model_name} using {method}.", styles["CustomBody"]))
                elements.append(Spacer(1, 6))

    # HPO CSV Reference
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("HPO Results Summary CSV", styles["CustomHeading"]))
    elements.append(Paragraph(
        f"The complete results of hyperparameter optimisation have been saved as a .csv file.<br/>"
        f"<b>Location:</b> {csv_path}", styles["CustomBody"]
    ))
    elements.append(Spacer(1, 12))

    # Best Models Summary Table
    elements.append(Paragraph("Best Models per Target Variable", styles["CustomHeading"]))

    if not best_models_per_target.empty:
        df = best_models_per_target.copy()

        # Normalise names and remove semantic duplicates (case-/underscore-/space-insensitive)
        def _norm(name: str) -> str:
            s = str(name).replace("\xa0", " ").replace("_", " ")
            s = " ".join(s.split())  # collapse internal whitespace
            return s.strip().lower()

        seen, keep = set(), []
        for c in df.columns:
            nc = _norm(c)
            if nc in seen:
                continue
            seen.add(nc)
            keep.append(c)
        df = df.loc[:, keep]

        # Rename to consistent display names
        df = df.rename(columns={
            "model_name": "Model Name",
            "target_variable": "Target Variable",
            "Target_Variable": "Target Variable",  # The table breaks if I don't include this, don't remove
            "hpo_method": "HPO Method",
            "hyperparameters": "Best Hyperparameters",
            "elapsed_time": "Elapsed Time (s)",
            "elapsed_time_s": "Elapsed Time (s)",
            "Elapsed Time (s)": "Elapsed Time (s)",
        })

        # Run the de-dupe pass again post-rename in case the rename created a clash
        seen, keep = set(), []
        for c in df.columns:
            nc = _norm(c)
            if nc in seen:
                continue
            seen.add(nc)
            keep.append(c)
        df = df.loc[:, keep]

        # Detect metric column dynamically
        metric_col = None
        for col in df.columns:
            if col.upper() in {"MSE", "MAE", "R^2", "Q^2"}:
                metric_col = col
                break
        metric_display_name = None
        if metric_col:
            metric_display_name = (metric_col.replace("^2", "²")
                                            .replace("R2", "R²")
                                            .replace("Q2", "Q²"))
            df = df.rename(columns={metric_col: metric_display_name})

        # Order columns
        col_order = ["Target Variable", "Model Name", "HPO Method", "Best Hyperparameters"]
        if metric_display_name:
            col_order.append(metric_display_name)
        if "Elapsed Time (s)" in df.columns:
            col_order.append("Elapsed Time (s)")
        df = df[[c for c in col_order if c in df.columns]]

        # Styles (wrap headers and body)
        header_style = ParagraphStyle(
            name="HeaderWrapped", fontName="Helvetica-Bold",
            fontSize=7, leading=8, textColor=colors.whitesmoke,
            wordWrap="CJK"
        )
        cell_style = ParagraphStyle(
            name="WrappedCell", fontSize=6.5, leading=7.5, wordWrap="CJK"
        )

        # Build table data with wrapped headers and cells
        header_row = [Paragraph(str(h), header_style) for h in df.columns]
        table_data = [header_row]

        for _, row in df.iterrows():
            row_cells = []
            for col, val in row.items():
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                row_cells.append(Paragraph(val_str, cell_style))
            table_data.append(row_cells)

        # Auto-fit width
        total_page_width = 175 * mm
        n_cols = len(df.columns)
        col_widths = [total_page_width / n_cols] * n_cols

        # Table
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            # Header text colour is already set in header_style; keep for non-Paragraph fallbacks:
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 12))

def add_postprocessing_section(
    elements,
    styles,
    postprocessing_results: dict,
    image_output_dir: str = "report_images"
):
    os.makedirs(image_output_dir, exist_ok=True)

    # === 1. Intro ===
    elements.append(Paragraph("Postprocessing and Residual Analysis", styles["CustomHeading"]))
    intro_text = (
        "This section analyses the quality of the best-performing models through residual-based diagnostics. "
        "These include cross-validation, outlier influence analysis (Cook's Distance), and residual transformations "
        "to assess assumptions such as homoscedasticity and normality."
    )
    elements.append(Paragraph(intro_text, styles["CustomBody"]))
    elements.append(Spacer(1, 12))

    # === 2. Cross-Validation Summary Table ===
    cv_df = postprocessing_results.get("cv_summary_df")
    scoring_metric = postprocessing_results.get("scoring_metric", "R²")  # new addition

    if cv_df is not None and not cv_df.empty:
        elements.append(Paragraph("Cross-Validation Results", styles["CustomSubheading"]))
        elements.append(Paragraph(
            f"The table below shows cross-validation scores for each target variable using "
            f"the selected CV method and <b>{scoring_metric}</b> as the evaluation metric.",
            styles["CustomBody"]
        ))

        # Round numeric columns for readability
        cv_df["Mean Score"] = cv_df["Mean Score"].astype(float).round(4)
        cv_df["Std Deviation"] = cv_df["Std Deviation"].astype(float).round(4)

        # Convert long dict string to wrapped Paragraph for CV Parameters
        wrapped_rows = []
        for row in cv_df.itertuples(index=False):
            wrapped_row = [
                str(row[0]),  # Target Variable
                str(row[1]),  # CV Method
                Paragraph(str(row[2]).replace(",", ",<br/>"), styles["CustomBody"]),  # CV Parameters with line breaks
                f"{row[3]:.4f}",  # Mean Score
                f"{row[4]:.4f}"   # Std Deviation
            ]
            wrapped_rows.append(wrapped_row)

        # Define column widths and build table
        table_data = [list(cv_df.columns)] + wrapped_rows
        col_widths = [40*mm, 30*mm, 55*mm, 25*mm, 25*mm]

        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#555555")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 12))

    # === 3. Cook’s Distance Plot ===
    cooks_fig = postprocessing_results.get("cooks_fig")
    if cooks_fig:
        path = os.path.join(image_output_dir, "cooks_distance.png")
        cooks_fig.savefig(path, dpi=300, bbox_inches='tight')
        elements.append(Paragraph("Cook's Distance", styles["CustomSubheading"]))
        elements.append(Paragraph(
            "Cook's Distance identifies influential data points in the test set that may disproportionately affect the model's predictions. "
            "Values exceeding the threshold (4/n) are flagged as potentially problematic.",
            styles["CustomBody"]
        ))
        elements.append(Image(path, width=14*cm, height=7*cm))
        elements.append(Spacer(1, 12))

    # === 4. Residuals with Influential Points ===
    residuals_fig = postprocessing_results.get("residuals_fig")
    if residuals_fig:
        path = os.path.join(image_output_dir, "residuals_with_influential.png")
        residuals_fig.savefig(path, dpi=300, bbox_inches='tight')
        elements.append(Paragraph("Residuals with Influential Points", styles["CustomSubheading"]))
        elements.append(Paragraph(
            "These plots visualise residuals versus predicted values. Points identified as influential (via Cook's Distance) are highlighted in red. "
            "Ideally, residuals should be symmetrically distributed around zero with no obvious patterns.",
            styles["CustomBody"]
        ))
        elements.append(Image(path, width=14*cm, height=7*cm))
        elements.append(Spacer(1, 12))

    # === 5. Residual Transformation Table (with AD highlighting) ===
    transformation_df = postprocessing_results.get("transformation_df", pd.DataFrame())
    if not transformation_df.empty:
        elements.append(Paragraph("Residual Transformation Summary", styles["CustomSubheading"]))
        explanation = (
            "To assess and potentially improve the normality of residuals, various transformations (Log, Sqrt, Box-Cox, Yeo-Johnson) "
            "were applied. The table below reports the resulting skewness, excess kurtosis, and Anderson-Darling (AD) statistic "
            "for each transformation and target. Lower values typically indicate improved normality. "
            "<br/><b>The row highlighted in light green represents the lowest AD Statistic per target variable.</b>"
        )
        elements.append(Paragraph(explanation, styles["CustomBody"]))
        elements.append(Spacer(1, 6))

        # Round numeric columns
        float_cols = ["Skewness", "Excess Kurtosis", "AD Statistic"]
        transformation_df[float_cols] = transformation_df[float_cols].round(4)

        # Identify minimum AD per target
        highlight_rows = set(
            transformation_df.loc[transformation_df.groupby("Target Variable")["AD Statistic"].idxmin()].index
        )

        table_data = [transformation_df.columns.tolist()] + transformation_df.values.tolist()
        table_data = [[str(item) for item in row] for row in table_data]

        # Table creation
        table = Table(table_data, repeatRows=1, hAlign='LEFT',
                      colWidths=[80, 80, 80, 60, 60, 60])

        style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#d9d9d9")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]

        # Highlight best AD statistic per target
        for row_idx in highlight_rows:
            style_commands.append(('BACKGROUND', (0, row_idx + 1), (-1, row_idx + 1), colors.lightgreen))

        table.setStyle(TableStyle(style_commands))
        elements.append(table)
        elements.append(Spacer(1, 12))
    else:
        elements.append(Paragraph("No residual transformation results available.", styles["CustomBody"]))

    # === 6. Transformed Residual Diagnostic Plots ===
    trans_figs = postprocessing_results.get("transformation_figs", {})
    if trans_figs:
        elements.append(Paragraph("Transformed Residual Diagnostic Plots", styles["CustomSubheading"]))
        transform_text = (
            "Residual transformations aim to correct skewness and non-normality, improving the validity of statistical assumptions. "
            "Each target variable is transformed using multiple methods (log, sqrt, Box-Cox, Yeo-Johnson), and the best is selected "
            "based on the Anderson-Darling test."
        )
        elements.append(Paragraph(transform_text, styles["CustomBody"]))
        elements.append(Spacer(1, 6))

        for name, fig in trans_figs.items():
            if fig:
                filename = f"transformed_{name}.png"
                path = os.path.join(image_output_dir, filename)
                fig.savefig(path, dpi=300, bbox_inches='tight')

                title_map = {
                    "residual": "Residuals vs. Predicted (Transformed)",
                    "histogram": "Histogram of Transformed Residuals",
                    "qq": "Q-Q Plot of Transformed Residuals"
                }
                caption_map = {
                    "residual": "Visual inspection of residuals after transformation. Good models show no structure and scatter around zero.",
                    "histogram": "Histogram overlaid with a normal distribution. Closer fit indicates improved normality.",
                    "qq": "Q-Q plot comparing transformed residuals against theoretical quantiles. A straight line implies normality."
                }

                elements.append(Paragraph(title_map.get(name, name), styles["CustomBody"]))
                elements.append(Paragraph(caption_map.get(name, ""), styles["CustomBody"]))
                elements.append(Image(path, width=14*cm, height=7*cm))
                elements.append(Spacer(1, 12))

    elements.append(Spacer(1, 24))

def add_artifacts_section(elements, styles, save_paths, models_dir):
    elements.append(Paragraph("Saved Models & Artifacts", styles["CustomHeading"]))
    elements.append(Paragraph(
        "The following files were saved during the run. Paths are absolute for reproducibility.",
        styles["CustomBody"]
    ))
    elements.append(Spacer(1, 8))

    # Models directory
    elements.append(Paragraph(f"<b>Models directory:</b> {models_dir}", styles["CustomBody"]))
    elements.append(Spacer(1, 6))

    # Per-target pipelines
    if "by_target" in save_paths and save_paths["by_target"]:
        elements.append(Paragraph("<b>Per-target pipelines (.pkl):</b>", styles["CustomBody"]))
        for tgt, pth in save_paths["by_target"].items():
            elements.append(Paragraph(f"• {tgt}: {pth}", styles["CustomBody"]))
        elements.append(Spacer(1, 6))

    # Metadata
    if "metadata" in save_paths:
        elements.append(Paragraph(f"<b>Metadata (.json):</b> {save_paths['metadata']}", styles["CustomBody"]))
        elements.append(Spacer(1, 4))

    # Bundle
    if "bundle" in save_paths:
        elements.append(Paragraph(f"<b>Bundle (.pkl):</b> {save_paths['bundle']}", styles["CustomBody"]))
        elements.append(Spacer(1, 6))
