# report_generation.py
# This module takes all the information gathered during the workflow and generates a .pdf report with all the information.
# I decided to add this as there was a lot of information being processed and it's nice for reproducibility and organisation.
# Please note that reports can reach multiple dozens of pages long and contain a lot of images. Some of these images can be hard to read, I'm working on it.
# For more information refer to the reportlab documentation.

import math
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from datetime import datetime
from reportlab.platypus import Image
import matplotlib.pyplot as plt

_PAGE_IMG_WIDTH  = 170 * mm   # usable width on an A4 page with 20mm margins
# A4 frame height with 20 mm top+bottom margins ≈ 789 pt; keep a safe margin below that.
_PAGE_IMG_HEIGHT = 230 * mm   # ~652 pt — one safe page-slice height for tall figures


def _proportional_image(path, max_width=_PAGE_IMG_WIDTH, max_height=_PAGE_IMG_HEIGHT):
    """Return a single ReportLab Image scaled to fit within max_width × max_height.
    Use _insert_image() instead when the image might need to span multiple pages.
    """
    img = Image(path)
    if not img.drawWidth:
        return Image(path, width=max_width)
    aspect = img.drawHeight / img.drawWidth
    w, h = max_width, max_width * aspect
    if max_height and h > max_height:
        h = max_height
        w = h / aspect
    return Image(path, width=w, height=h)


def _insert_image(elements, path, max_width=_PAGE_IMG_WIDTH, max_height=_PAGE_IMG_HEIGHT,
                  n_subplot_rows=None):
    """Insert an image into elements, slicing it into page-height strips if too tall.

    When n_subplot_rows is provided, cuts are made only at subplot-row boundaries so no
    individual subplot panel is split across two pages.  A CondPageBreak is placed before
    every strip so each strip lands on a page with enough room to hold it fully.
    Falls back to a single scaled Image if PIL cannot open the file.
    """
    from reportlab.platypus import CondPageBreak

    try:
        pil_img = PILImage.open(path)
        pil_img.load()
        img_w_px, img_h_px = pil_img.size
    except Exception:
        elements.append(_proportional_image(path, max_width, max_height))
        return

    if img_w_px == 0 or img_h_px == 0:
        pil_img.close()
        return

    pts_per_pixel = max_width / img_w_px
    display_h = img_h_px * pts_per_pixel

    if display_h <= max_height:
        # Fits on one page — CondPageBreak prevents it starting near the bottom
        elements.append(CondPageBreak(display_h))
        elements.append(Image(path, width=max_width, height=display_h))
        pil_img.close()
        return

    # Build y-cut boundaries at subplot-row boundaries when possible ----------------
    if n_subplot_rows and n_subplot_rows > 1:
        row_h_px   = img_h_px / n_subplot_rows
        row_h_pts  = row_h_px * pts_per_pixel
        rows_per_slice = max(1, int(max_height / row_h_pts))
        y_cuts = [int(round(r * row_h_px))
                  for r in range(0, n_subplot_rows, rows_per_slice)]
        y_cuts.append(img_h_px)
        y_cuts = sorted(set(y_cuts))
    else:
        # No row metadata — fall back to plain pixel slicing
        pixels_per_slice = max(1, int(max_height / pts_per_pixel))
        y_cuts = list(range(0, img_h_px, pixels_per_slice)) + [img_h_px]

    base, ext = os.path.splitext(path)

    for i, (y0, y1) in enumerate(zip(y_cuts[:-1], y_cuts[1:])):
        strip = pil_img.crop((0, y0, img_w_px, y1))
        strip_path = f"{base}_part{i + 1}{ext}"
        strip.save(strip_path, format="PNG")
        strip.close()

        strip_display_h = (y1 - y0) * pts_per_pixel
        elements.append(CondPageBreak(strip_display_h))
        elements.append(Image(strip_path, width=max_width, height=strip_display_h))
        if i < len(y_cuts) - 2:
            elements.append(Spacer(1, 4))

    pil_img.close()

import os
import pandas as pd

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
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plot_paths[label] = path
        plt.close(fig)

    return plot_paths


def _build_dist_corr_caption(dummy, mp):
    """Return a list of paragraph strings for the Distance Correlation report caption."""
    paras = [
        "Distance correlation matrix across input features. Distance correlation captures general "
        "(including non-linear) dependence; values near 1 suggest strong dependence, while values "
        "near 0 suggest weak or no dependence. Use this to spot redundant features or feature clusters."
    ]
    if dummy:
        paras.append(
            "A dummy (purely random) variable is included as a noise baseline. Any feature whose "
            "distance correlation with another variable is similar to or below the dummy's row of values "
            "is exhibiting dependence no stronger than random noise, and may not carry genuine information."
        )
    if mp:
        paras.append(
            "Marchenko-Pastur (MP) thresholding has been applied to denoise the matrix. This technique "
            "uses Random Matrix Theory to estimate how many independent correlation patterns in the data "
            "are genuinely stronger than what would arise by chance in a matrix of the same size. Patterns "
            "weaker than this noise floor are discarded, and the matrix is reconstructed from the remaining "
            "signal patterns only."
        )
        paras.append(
            "The noise threshold lambda+ = (1 + sqrt(p/n))^2 is the Marchenko-Pastur upper bound, where "
            "p is the number of features and n the number of samples. Any eigenvalue of the correlation "
            "matrix below lambda+ is statistically indistinguishable from random noise at the given "
            "dataset size and is discarded. A larger lambda+ (when p is large relative to n) means the "
            "bar for retaining a signal component is higher, so fewer components survive."
        )
        paras.append(
            "The subtitle on the plot shows how many signal components were retained. When only a few "
            "are kept, the dataset's correlation structure is dominated by those shared patterns. "
            "Features with high values in the denoised matrix are strongly tied to the retained signal; "
            "features near zero after denoising show correlations no stronger than sampling noise."
        )
    return paras


def add_preprocessing_section(elements, results, plot_paths, dataset_path, styles,
                              dist_corr_dummy=True, dist_corr_mp=False):
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
        ["Feature scaling", meta.get("scaler_name", "StandardScaler")],
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
        "Distance Correlation": _build_dist_corr_caption(dist_corr_dummy, dist_corr_mp),
    }

    # Render plots with their richer captions
    for label, img_path in plot_paths.items():
        if not os.path.exists(img_path):
            continue

        # Pick caption — handle multi-part keys like "Boxplots Part 2" or "Features vs CO Part 2"
        if label.startswith("Features vs "):
            caption = captions["_features_vs_"]
        elif label.startswith("Boxplots"):
            caption = captions["Boxplots"]
        else:
            caption = captions.get(label, label)

        if isinstance(caption, list):
            for para in caption:
                elements.append(Paragraph(para, styles["CustomBody"]))
                elements.append(Spacer(1, 5))
        else:
            elements.append(Paragraph(caption, styles["CustomBody"]))
        _insert_image(elements, img_path)
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

    # Convert the DataFrame to a list of lists (rows) — round numeric columns only
    num_cols = display_df.select_dtypes(include='number').columns
    display_df[num_cols] = display_df[num_cols].round(4)
    data = [display_df.columns.tolist()] + display_df.values.tolist()

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

        n_jobs = uq_settings.get("n_jobs", 1)
        settings_html = f"""
            <b>UQ Method:</b> {method_used}<br/>
            <b>Number of Bootstraps:</b> {n_bootstrap}<br/>
            <b>Confidence Interval (CI):</b> {conf_interval}%<br/>
            <b>Calibration Fraction:</b> {calibration_frac}<br/>
            <b>Subsample Test Size:</b> {test_size}<br/>
            <b>Parallel Jobs (Bootstrap):</b> {n_jobs}
        """
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(settings_html, styles["CustomBody"]))

    elements.append(Spacer(1, 12))

    # Add plots
    for label, img_path in uq_plot_paths.items():
        if os.path.exists(img_path):
            elements.append(Paragraph(label, styles["CustomBody"]))
            _insert_image(elements, img_path)
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
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
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

    # Plot type descriptions — keyed by the prefix used in figure names
    descriptions = {
        "ICE_PDP":        "ICE and PDP plots show how individual features affect model predictions. ICE (blue) shows per-sample curves, while PDP (orange) shows the average trend.",
        "SHAP_Summary":   "SHAP summary plots show the overall importance and distribution of SHAP values across all features.",
        "SHAP_Dependence":"SHAP dependence plots illustrate how each feature's value relates to its SHAP value, indicating feature interaction and impact.",
    }
    _PREFIXES = ("ICE_PDP_", "SHAP_Summary_", "SHAP_Dependence_")

    for fig_name, fig_or_list in interpretability_figures.items():
        # plotting functions return a list of figures (one per PDF page worth of rows)
        fig_list = fig_or_list if isinstance(fig_or_list, list) else [fig_or_list]

        # Extract label type and target — show caption once before the first part
        label_type = fig_name
        target_name = ""
        for prefix in _PREFIXES:
            if fig_name.startswith(prefix):
                label_type = prefix.rstrip("_")
                target_name = fig_name[len(prefix):]
                break

        display_label = label_type.replace("_", " ")
        caption = f"{display_label} for {target_name}" if target_name else display_label
        elements.append(Paragraph(caption, styles["CustomBody"]))
        elements.append(Paragraph(descriptions.get(label_type, "Interpretability plot."), styles["CustomBody"]))

        for part_idx, fig in enumerate(fig_list):
            part_suffix = f"_part{part_idx + 1}" if len(fig_list) > 1 else ""
            fig_path = os.path.join(output_dir, f"{fig_name.replace(' ', '_')}{part_suffix}.png")
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            _insert_image(elements, fig_path)
            if part_idx < len(fig_list) - 1:
                elements.append(Spacer(1, 4))

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
    output_dir: str = "report_images",
    early_stopping: dict = None,
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

    # Build early-stopping description per method
    es = early_stopping or {}
    es_method_map = {"random": "random_search", "hyperopt": "hyperopt", "skopt": "skopt"}
    def _es_desc(method_key):
        cfg = es.get(es_method_map.get(method_key, ""), {})
        p = cfg.get("patience")
        d = cfg.get("min_delta", 1e-4)
        if p is None:
            return "early stopping disabled"
        return f"early stopping: patience={p}, min_delta={d}"

    settings_text = "This section presents the best performance for each model and target variable across the selected HPO methods. "
    settings_text += f"The chosen evaluation metric is <b>{metric_used}</b>.<br/><br/>"

    settings_text += "<b>HPO Methods and Settings:</b><br/>"
    for method in methods_used:
        if method in methods_description:
            settings_text += f"• <b>{method.capitalize()}</b>: {methods_description[method]}; {_es_desc(method)}<br/>"

    elements.append(Paragraph(settings_text, styles["CustomBody"]))
    elements.append(Spacer(1, 12))

    # Per-model breakdown — only iterate methods that actually produced results
    active_methods = [m for m in methods_used if m in hpo_metrics and hpo_metrics[m]]
    if not active_methods:
        return
    model_names = list(hpo_metrics[active_methods[0]].keys())

    for model_name in model_names:
        elements.append(Paragraph(f"{model_name}", styles["CustomSubheading"]))

        for method in active_methods:
            if model_name not in hpo_metrics[method]:
                continue

            elements.append(Paragraph(f"<b>Method:</b> {method.capitalize()}", styles["CustomBody"]))

            # Table Header
            table_data = [["Target Variable", f"Best {metric_used}", "Iterations", "Elapsed Time (s)", "Best Parameters"]]

            for target_var, metric_data in hpo_metrics[method][model_name].items():
                best_score = metric_data.get(metric_used)
                elapsed = round(metric_data.get("elapsed_time", 0), 2)
                actual = metric_data.get("actual_iters")
                max_it = metric_data.get("max_iters")
                stopped = metric_data.get("stopped_early", False)
                if actual is not None and max_it is not None:
                    iter_str = f"{actual}/{max_it} (ES)" if stopped else f"{actual}/{max_it}"
                else:
                    iter_str = "N/A"
                best_params = hpo_params[method][model_name].get(target_var, {})
                params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
                params_paragraph = Paragraph(params_str if params_str else "N/A", param_style)

                table_data.append([
                    target_var,
                    f"{best_score:.4f}" if best_score is not None else "N/A",
                    iter_str,
                    f"{elapsed:.2f}",
                    params_paragraph
                ])

            # Style and insert table
            table = Table(table_data, colWidths=[35*mm, 25*mm, 25*mm, 25*mm, 60*mm])
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
                elements.append(_proportional_image(plot_path, max_width=14*cm, max_height=10*cm))
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

def add_pareto_section(elements, styles, plot_paths: dict, perf_metric: str):
    """
    Add a Pareto front analysis section to the report — one figure per target variable.
    plot_paths: {target_var: image_path} from save_pareto_plots().
    """
    if not plot_paths:
        return

    elements.append(Paragraph("Pareto Front Analysis", styles["CustomHeading"]))
    elements.append(Paragraph(
        "The Pareto front identifies models where no alternative offers both better performance "
        "and lower training cost simultaneously. Orange points are Pareto-optimal — removing any "
        "one of them would require accepting a worse trade-off. Blue points are dominated: at least "
        "one other model achieves equal or better performance in equal or less training time.",
        styles["CustomBody"]
    ))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(
        f"Performance axis shows the best {perf_metric} achieved across all HPO methods. "
        "Training time is the wall-clock time recorded during baseline model training. "
        "Axes switch to log scale automatically when values span more than one order of magnitude.",
        styles["CustomBody"]
    ))
    elements.append(Spacer(1, 8))

    for target_var, img_path in plot_paths.items():
        if not os.path.exists(img_path):
            continue
        _insert_image(elements, img_path)
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
        cooks_fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(cooks_fig)
        elements.append(Paragraph("Cook's Distance", styles["CustomSubheading"]))
        elements.append(Paragraph(
            "Cook's Distance identifies influential data points in the test set that may disproportionately affect the model's predictions. "
            "Values exceeding the threshold (4/n) are flagged as potentially problematic.",
            styles["CustomBody"]
        ))
        elements.append(_proportional_image(path, max_width=14*cm, max_height=10*cm))
        elements.append(Spacer(1, 12))

    # === 4. Residuals with Influential Points ===
    residuals_fig = postprocessing_results.get("residuals_fig")
    if residuals_fig:
        path = os.path.join(image_output_dir, "residuals_with_influential.png")
        residuals_fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(residuals_fig)
        elements.append(Paragraph("Residuals with Influential Points", styles["CustomSubheading"]))
        elements.append(Paragraph(
            "These plots visualise residuals versus predicted values. Points identified as influential (via Cook's Distance) are highlighted in red. "
            "Ideally, residuals should be symmetrically distributed around zero with no obvious patterns.",
            styles["CustomBody"]
        ))
        elements.append(_proportional_image(path, max_width=14*cm, max_height=10*cm))
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

        # Round numeric columns (coerce None/NaN to float first to avoid TypeError)
        float_cols = ["Skewness", "Excess Kurtosis", "AD Statistic"]
        transformation_df[float_cols] = transformation_df[float_cols].apply(pd.to_numeric, errors="coerce").round(4)

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
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)

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
                elements.append(_proportional_image(path, max_width=14*cm, max_height=10*cm))
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


def add_perl_section(elements, styles, perl_results: dict, perl_config: dict, output_dir: str):
    """Add a Physics-Enhanced Machine Learning (PERL) section to the PDF report.

    Includes: description, LaTeX equations, reconstruction map table,
    RMSE metrics table, and predicted-vs-actual scatter plots per target.
    """
    import io
    import numpy as np
    from matplotlib.figure import Figure

    os.makedirs(output_dir, exist_ok=True)

    elements.append(Spacer(1, 16))
    elements.append(Paragraph("Physics-Enhanced Machine Learning (PERL) Reconstruction",
                               styles["CustomHeading"]))
    elements.append(Paragraph(
        "Physics-Enhanced Residual Learning (PERL) combines first-principles physics equations with a "
        "data-driven ML model. First, the physics equations are applied to the inputs to produce a "
        "structured estimate of each target variable. The ML model is then trained only on the "
        "<i>residuals</i> — the gap between that physics estimate and what was actually measured. "
        "At prediction time, the physics estimate and the ML residual correction are simply added "
        "together to give the final output. This means the ML model has a much smaller, better-defined "
        "problem to solve, and the physics knowledge acts as an anchor that keeps predictions "
        "physically sensible even outside the training range.",
        styles["CustomBody"],
    ))
    elements.append(Spacer(1, 6))

    # ── 0. Configuration info block ───────────────────────────────────────────
    mode = perl_config.get("mode", "expression")
    config_path = perl_config.get("_config_path", "")
    mode_label = "Expression Mode" if mode == "expression" else "Script Mode"
    config_info = f"<b>Physics mode:</b> {mode_label}"
    if config_path:
        config_info += f"<br/><b>Physics configuration file:</b> {config_path}"
    if mode == "script":
        script_path = perl_config.get("script_path", "")
        if script_path:
            config_info += f"<br/><b>Physics script:</b> {script_path}"
    elements.append(Paragraph(config_info, styles["CustomBody"]))
    elements.append(Spacer(1, 10))

    # ── 1. Physics expressions as LaTeX images ───────────────────────────────
    expressions = perl_config.get("expressions", [])
    if mode == "script":
        # Script mode: governing equations live in the script file — cite it instead
        script_path = perl_config.get("script_path", "")
        elements.append(Paragraph("Physics Model", styles["CustomSubheading"]))
        elements.append(Paragraph(
            "This run used Script Mode. The governing equations are defined in the "
            f"<b>governing_function</b> of the physics script cited above"
            + (f": <i>{script_path}</i>" if script_path else "") + ".",
            styles["CustomBody"],
        ))
        elements.append(Spacer(1, 8))
    elif expressions:
        elements.append(Paragraph("Physics Expressions", styles["CustomSubheading"]))
        elements.append(Paragraph(
            "The following expressions were used to compute physics-based estimates and residuals:",
            styles["CustomBody"],
        ))
        elements.append(Spacer(1, 4))

        try:
            from phoenix_ml.physics_expressions import parse_expression, expression_to_latex
        except ImportError:
            parse_expression = expression_to_latex = None

        for i, expr_text in enumerate(expressions, start=1):
            expr_text = expr_text.strip()
            if not expr_text:
                continue

            # Try to render as LaTeX; fall back to plain text
            latex_img_path = None
            if parse_expression and expression_to_latex:
                try:
                    lhs, tree, name_map = parse_expression(expr_text)
                    latex_str = expression_to_latex(lhs, tree, name_map)
                    fig = Figure(figsize=(7, 0.5), dpi=150)
                    fig.patch.set_facecolor("white")
                    fig.text(0.01, 0.5, f"${latex_str}$",
                             fontsize=13, va="center", ha="left", color="black")
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight",
                                pad_inches=0.1, facecolor="white")
                    buf.seek(0)
                    pil_img = PILImage.open(buf)
                    pil_img.load()
                    latex_img_path = os.path.join(output_dir, f"perl_expr_{i}.png")
                    pil_img.save(latex_img_path)
                except Exception:
                    latex_img_path = None

            if latex_img_path and os.path.isfile(latex_img_path):
                elements.append(_proportional_image(latex_img_path, max_width=160 * mm, max_height=25 * mm))
            else:
                elements.append(Paragraph(f"[{i}]  {expr_text}", styles["CustomBody"]))

        elements.append(Spacer(1, 8))

    # ── 2. Reconstruction map table ──────────────────────────────────────────
    recon_map = perl_config.get("reconstruction_map", {})
    if recon_map:
        elements.append(Paragraph("Reconstruction Map", styles["CustomSubheading"]))
        elements.append(Paragraph(
            "Each residual target is paired with its physics estimate column for reconstruction:",
            styles["CustomBody"],
        ))
        tbl_data = [["Residual Target", "Physics Estimate Column"]]
        for tgt, phys_col in recon_map.items():
            tbl_data.append([tgt, phys_col])
        tbl = Table(tbl_data, hAlign="LEFT", colWidths=[80 * mm, 80 * mm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
            ("GRID",       (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
        ]))
        elements.append(tbl)
        elements.append(Spacer(1, 10))

    # ── 3. RMSE metrics table ────────────────────────────────────────────────
    elements.append(Paragraph("PERL Reconstruction Metrics", styles["CustomSubheading"]))
    has_comparison = any(
        not (isinstance(v.get("rmse_physics"), float) and np.isnan(v["rmse_physics"]))
        for v in perl_results.values()
    )
    if has_comparison:
        hdr = ["Target Variable", "Physics-only RMSE", "PERL RMSE", "Improvement (%)"]
        col_w = [60 * mm, 35 * mm, 35 * mm, 35 * mm]
    else:
        hdr = ["Target Variable", "ML Residual RMSE"]
        col_w = [90 * mm, 70 * mm]

    metrics_data = [hdr]
    for target, res in perl_results.items():
        rmse_phys = res.get("rmse_physics", float("nan"))
        rmse_perl = res.get("rmse_perl",    float("nan"))
        rmse_ml   = res.get("rmse_ml_residual", float("nan"))
        if has_comparison:
            if not np.isnan(rmse_phys) and not np.isnan(rmse_perl) and rmse_phys > 0:
                improvement = f"{(rmse_phys - rmse_perl) / rmse_phys * 100:+.1f}%"
            else:
                improvement = "n/a"
            metrics_data.append([
                target,
                f"{rmse_phys:.4f}" if not np.isnan(rmse_phys) else "n/a",
                f"{rmse_perl:.4f}" if not np.isnan(rmse_perl) else "n/a",
                improvement,
            ])
        else:
            metrics_data.append([target, f"{rmse_ml:.4f}" if not np.isnan(rmse_ml) else "n/a"])

    metrics_tbl = Table(metrics_data, hAlign="LEFT", colWidths=col_w)
    metrics_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
    ]))
    elements.append(metrics_tbl)
    elements.append(Spacer(1, 12))

    # ── 4. Predicted-vs-actual scatter plots (one figure per target) ─────────
    plot_targets = [t for t, v in perl_results.items() if v.get("y_actual") is not None]
    if plot_targets:
        elements.append(Paragraph("Predicted vs. Actual Plots", styles["CustomSubheading"]))
        elements.append(Paragraph(
            "Each figure compares the physics-only estimate (left) and the full PERL prediction "
            "(right) against the measured values. A perfect model lies on the diagonal dashed line.",
            styles["CustomBody"],
        ))
        elements.append(Spacer(1, 6))

    for target in plot_targets:
        res = perl_results[target]
        y_actual  = np.asarray(res["y_actual"])
        y_physics = np.asarray(res["y_physics"])
        y_perl    = np.asarray(res["y_perl"])
        rmse_phys     = res.get("rmse_physics", float("nan"))
        rmse_perl_val = res.get("rmse_perl",    float("nan"))
        original_col  = res.get("original_col") or target

        if len(y_actual) == 0 or len(y_physics) == 0:
            elements.append(Paragraph(f"No data available to plot for {target}.", styles["CustomBody"]))
            continue

        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(f"PERL Reconstruction — {original_col}", fontsize=11, fontweight="bold")

            for ax, y_pred, label, rmse_val, colour in [
                (axes[0], y_physics, "Physics Estimate", rmse_phys,    "#2980b9"),
                (axes[1], y_perl,    "PERL Prediction",  rmse_perl_val, "#27ae60"),
            ]:
                lo = min(float(y_actual.min()), float(y_pred.min()))
                hi = max(float(y_actual.max()), float(y_pred.max()))
                margin = (hi - lo) * 0.05 if hi != lo else abs(hi) * 0.05 + 1e-6
                ax.scatter(y_actual, y_pred, alpha=0.45, s=14, color=colour, edgecolors="none")
                ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                        "k--", linewidth=1.0)
                ax.set_xlabel(f"Actual {original_col}", fontsize=8)
                ax.set_ylabel(label, fontsize=8)
                rmse_str = f"{rmse_val:.4f}" if not np.isnan(rmse_val) else "n/a"
                ax.set_title(f"{label}  (RMSE = {rmse_str})", fontsize=9)
                ax.tick_params(labelsize=7)
                ax.set_xlim(lo - margin, hi + margin)
                ax.set_ylim(lo - margin, hi + margin)

            fig.tight_layout()
            plot_path = os.path.join(output_dir, f"perl_scatter_{target}.png")
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            elements.append(_proportional_image(plot_path, max_width=160 * mm))
            elements.append(Spacer(1, 10))
        except Exception as exc:
            plt.close("all")
            elements.append(Paragraph(f"Could not generate plot for {target}: {exc}", styles["CustomBody"]))
