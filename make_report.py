"""Generate the two-page, two-column AI-A2 descriptive report.

Run from the project root: python make_report.py
"""
from pathlib import Path
import json
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "AI_A2_Report.pdf"
metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
sensitivity_path = RESULTS / "reward_sensitivity.json"
if not sensitivity_path.exists():
    raise FileNotFoundError("Reward sensitivity results are missing. Run `python run_reward_sensitivity.py` before generating the report.")
sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))

# Team details and repository URL for this submission.
TEAM_MEMBERS = "Shahid Samir Khatib - B24CM1055"
GITHUB_URL = "https://github.com/Shahid20-K/AI-A2-Canteen-qlearning"

PAGE_W, PAGE_H = A4
MARGIN = 30
GUTTER = 18
COL_W = (PAGE_W - 2 * MARGIN - GUTTER) / 2
FULL_W = PAGE_W - 2 * MARGIN
NAVY = colors.HexColor("#172B4D")
BLUE = colors.HexColor("#245C9F")
PALE = colors.HexColor("#EAF1F8")
LIGHT = colors.HexColor("#F4F6F8")
MID = colors.HexColor("#526273")
DARK = colors.HexColor("#202A35")
BORDER = colors.HexColor("#CCD5DF")
WHITE = colors.white

styles = {
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=7.6, leading=9.8,
                            textColor=DARK, spaceAfter=3, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=6.8, leading=8.5,
                             textColor=DARK, spaceAfter=2),
    "tiny": ParagraphStyle("tiny", fontName="Helvetica", fontSize=6.2, leading=7.5,
                            textColor=MID, spaceAfter=1),
    "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=9.1,
                               leading=10.4, textColor=NAVY, spaceBefore=3, spaceAfter=4),
    "subsection": ParagraphStyle("subsection", fontName="Helvetica-Bold", fontSize=7.3,
                                  leading=8.6, textColor=BLUE, spaceBefore=2, spaceAfter=2),
    "equation": ParagraphStyle("equation", fontName="Helvetica-Bold", fontSize=8.1,
                                leading=10.5, textColor=NAVY, alignment=TA_CENTER,
                                spaceBefore=1, spaceAfter=1),
    "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=6.1,
                               leading=7.2, textColor=MID, alignment=TA_CENTER, spaceAfter=2),
}


def draw_header(c, page_num):
    c.setFillColor(NAVY)
    c.roundRect(MARGIN, PAGE_H - 77, FULL_W, 47, 5, fill=1, stroke=0)
    c.setFillColor(WHITE)
    if page_num == 1:
        title = "Adaptive Meal Preparation in a Campus Canteen"
        subtitle = "A tabular Q-learning formulation for uncertain student demand"
    else:
        title = "Experimental Results and Discussion"
        subtitle = "Synthetic simulation results, interpretation, and limitations"
    c.setFont("Helvetica-Bold", 13 if page_num == 1 else 12)
    c.drawString(MARGIN + 12, PAGE_H - 49, title)
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN + 12, PAGE_H - 65, subtitle)
    if page_num == 1:
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 6.8)
        c.drawString(MARGIN, PAGE_H - 91, "TEAM MEMBER:")
        c.setFont("Helvetica", 6.8)
        c.drawString(MARGIN + 69, PAGE_H - 91, TEAM_MEMBERS)
        c.setFont("Helvetica-Bold", 6.8)
        c.drawString(MARGIN, PAGE_H - 103, "GITHUB:")
        c.setFont("Helvetica", 6.8)
        c.drawString(MARGIN + 34, PAGE_H - 103, GITHUB_URL)
    else:
        c.setFillColor(MID)
        c.setFont("Helvetica", 6.5)
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - 91, "Synthetic simulation only; no real canteen records used.")
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.5)
    c.line(MARGIN, 24, PAGE_W - MARGIN, 24)
    c.setFillColor(MID)
    c.setFont("Helvetica", 6)
    c.drawString(MARGIN, 13, "AI Assignment 2 | Real-Life Use Cases of Reinforcement Learning")
    c.drawRightString(PAGE_W - MARGIN, 13, f"Page {page_num} of 2")


def paragraph(c, text, x, y_top, width, style="body", gap=0):
    p = Paragraph(text, styles[style])
    if style == "equation":
        # A one-cell table gives predictable background/border padding without overlapping
        # the preceding paragraph in the manually laid-out two-column report.
        t = Table([[p]], colWidths=[width], hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PALE),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        _, h = t.wrap(width, PAGE_H)
        t.drawOn(c, x, y_top - h)
    else:
        _, h = p.wrap(width, PAGE_H)
        p.drawOn(c, x, y_top - h)
    return y_top - h - gap


def section(c, title, x, y, width):
    y = paragraph(c, title, x, y, width, "section", gap=1)
    c.setStrokeColor(BLUE)
    c.setLineWidth(0.6)
    c.line(x, y + 1, x + width, y + 1)
    return y - 4


def bullet(c, text, x, y, width, style="body", gap=1):
    return paragraph(c, f'<font color="{BLUE.hexval()}">•</font> {text}', x + 2, y, width - 2, style, gap)


def draw_table(c, data, x, y_top, width, col_widths=None, font=6.4, header=True):
    table_data = []
    for r, row in enumerate(data):
        converted = []
        for val in row:
            if isinstance(val, str):
                sty = ParagraphStyle(
                    f"tbl_{r}_{len(converted)}", fontName="Helvetica-Bold" if (header and r == 0) else "Helvetica",
                    fontSize=font, leading=font + 1.5,
                    textColor=WHITE if (header and r == 0) else DARK,
                )
                converted.append(Paragraph(val, sty))
            else:
                converted.append(val)
        table_data.append(converted)
    t = Table(table_data, colWidths=col_widths, hAlign="LEFT")
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
    ]
    if header:
        cmds.extend([("BACKGROUND", (0, 0), (-1, 0), NAVY),
                     ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT])])
    else:
        cmds.append(("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT]))
    t.setStyle(TableStyle(cmds))
    _, h = t.wrap(width, PAGE_H)
    t.drawOn(c, x, y_top - h)
    return y_top - h - 5


def draw_rl_cycle(c, y_top):
    """Draw a compact full-width summary of the agent-environment interaction."""
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(MARGIN, y_top, "RL decision cycle")
    box_top = y_top - 9
    box_h = 31
    gap = 8
    box_w = (FULL_W - 4 * gap) / 5
    labels = ["State + forecast\nhistory", "Q-learning\nagent", "Prep action", "Canteen\nenvironment", "Reward + next\nstate"]
    for i, label in enumerate(labels):
        x = MARGIN + i * (box_w + gap)
        y = box_top - box_h
        c.setFillColor(PALE if i % 2 == 0 else LIGHT)
        c.setStrokeColor(BORDER)
        c.roundRect(x, y, box_w, box_h, 4, fill=1, stroke=1)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 6.7)
        lines = label.split("\n")
        start_y = y + box_h / 2 + (len(lines) - 1) * 3.5
        for j, line in enumerate(lines):
            c.drawCentredString(x + box_w / 2, start_y - j * 8, line)
        if i < len(labels) - 1:
            x1 = x + box_w + 1
            x2 = x + box_w + gap - 1
            ym = y + box_h / 2
            c.setStrokeColor(BLUE)
            c.setLineWidth(1)
            c.line(x1, ym, x2, ym)
            c.line(x2 - 3, ym + 2.5, x2, ym)
            c.line(x2 - 3, ym - 2.5, x2, ym)
    c.setFillColor(MID)
    c.setFont("Helvetica-Oblique", 6)
    c.drawString(MARGIN, box_top - box_h - 10,
                 "The next state and reward feed into the Q-value update; the loop repeats until the simulated day ends.")


def draw_page_one(c):
    draw_header(c, 1)
    top = PAGE_H - 117
    left_x = MARGIN
    right_x = MARGIN + COL_W + GUTTER
    yL = top
    yR = top

    yL = section(c, "1. Problem statement", left_x, yL, COL_W)
    yL = paragraph(c,
        "A campus canteen must prepare enough meals to meet changing student demand without preparing so much that perishable food is discarded. Fixed preparation plans may respond poorly to day-to-day demand changes. This project studies whether a learned policy can balance availability and waste in a controlled simulation.",
        left_x, yL, COL_W, "body", gap=2)

    yL = section(c, "2. Environment and demand", left_x, yL, COL_W)
    yL = paragraph(c,
        "One day has eight time slots with nominal mean demand <b>(8, 12, 22, 30, 24, 18, 27, 14)</b>. The hidden low/normal/high regime has multipliers 0.70/1.00/1.30 and prior probabilities 0.25/0.50/0.25. The agent sees a noisy category forecast: it is correct with probability 70%; when wrong, one of the other categories is selected uniformly. Demand is Poisson-distributed around the hidden regime-adjusted mean.",
        left_x, yL, COL_W, "body", gap=2)
    yL = paragraph(c,
        "The inventory capacity is 40 portions. Of unsold portions, 30% (rounded down) can carry into the next slot; the remainder is discarded. Nothing carries beyond the final slot. These are explicit synthetic assumptions, not measured canteen data.",
        left_x, yL, COL_W, "body", gap=2)
    yL = paragraph(c,
        "At each step, available stock equals starting inventory plus actual preparation, capped at capacity. Sales are the minimum of available stock and demand; unmet demand is the difference when demand exceeds stock. Carryover and discarded waste are then calculated before the next slot.",
        left_x, yL, COL_W, "small", gap=2)

    yL = section(c, "3. State space S", left_x, yL, COL_W)
    yL = paragraph(c,
        "The agent observes <b>(time slot, exact inventory, noisy daily outlook, sales pace, prior-stockout flag)</b>. Sales pace is below/on/above expected cumulative sales from completed slots; the stockout flag warns that low sales may reflect unavailable food. The true regime remains hidden, and this compact history summary is not a full belief state.",
        left_x, yL, COL_W, "body", gap=2)
    yL = paragraph(c, "State count = 8 × 41 × 3 × 3 × 2 = <b>5,904 discrete states</b>.", left_x, yL, COL_W, "equation", gap=2)
    yL = paragraph(c,
        "The sales-pace category compares cumulative sales with cumulative posterior-expected demand: below 85% is below expected, above 115% is above expected, otherwise sales are on track. A prior-stockout flag helps identify censored sales. Demand remains uncertain.",
        left_x, yL, COL_W, "small", gap=2)

    yL = section(c, "4. Action space A", left_x, yL, COL_W)
    yL = paragraph(c,
        "At each slot, the agent selects an additional preparation batch from <b>{0, 5, 10, 15, 20}</b> portions. The 20-portion per-slot maximum is an assumed batch-size limit, not a measured kitchen capacity. Separately, total inventory is capped at 40 portions.",
        left_x, yL, COL_W, "body", gap=2)
    yL = section(c, "5. Episode objective and hypothesis", left_x, yL, COL_W)
    yL = paragraph(c,
        "One episode is a complete simulated day of eight decisions; its return is the sum of slot rewards. The experiment tests whether a state-dependent learned policy can outperform static preparation rules when the daily demand outlook changes. The policy is not assumed to succeed: this claim is assessed using held-out simulated episodes and operational metrics.",
        left_x, yL, COL_W, "body", gap=2)
    yL = section(c, "6. Slot transition", left_x, yL, COL_W)
    yL = paragraph(c,
        "For slot t, available stock is A<sub>t</sub> = min(I<sub>t</sub> + p<sub>t</sub>, 40). Sales are min(A<sub>t</sub>, D<sub>t</sub>) and unmet demand is max(D<sub>t</sub> − A<sub>t</sub>, 0). Leftover stock is max(A<sub>t</sub> − D<sub>t</sub>, 0); carryover is floor(0.30 × leftover) except at the final slot, where it is zero. Discarded waste equals leftover minus carryover.",
        left_x, yL, COL_W, "small", gap=2)
    yL = paragraph(c,
        "The hidden regime persists for the day while the forecast is noisy. Sales pace and prior stockouts summarize earlier outcomes but do not form a full Bayesian belief state, so the process remains partially observable.",
        left_x, yL, COL_W, "small", gap=2)

    yR = section(c, "7. Reward function R", right_x, yR, COL_W)
    yR = paragraph(c,
        "The reward encourages serving demand while accounting for preparation, stockouts, and food discarded:",
        right_x, yR, COL_W, "body", gap=2)
    yR = paragraph(c,
        "r = sold − 0.35 × prepared − 1.00 × unmet demand − 1.50 × discarded waste",
        right_x, yR, COL_W, "equation", gap=3)
    yR = bullet(c, "<b>sold:</b> portions served to students.", right_x, yR, COL_W)
    yR = bullet(c, "<b>prepared:</b> additional portions actually produced in the slot.", right_x, yR, COL_W)
    yR = bullet(c, "<b>unmet demand:</b> demand not met because inventory was insufficient.", right_x, yR, COL_W)
    yR = bullet(c, "<b>discarded waste:</b> unsold portions that do not carry over.", right_x, yR, COL_W, gap=2)
    yR = paragraph(c,
        "The coefficients are illustrative utility weights, not monetary estimates. They make competing objectives explicit. Changing these weights can change the policy, so they should be reported and justified rather than tuned invisibly.",
        right_x, yR, COL_W, "small", gap=2)

    yR = section(c, "8. Learning algorithm", right_x, yR, COL_W)
    yR = paragraph(c,
        "We use tabular Q-learning with epsilon-greedy exploration. The update is:",
        right_x, yR, COL_W, "body", gap=2)
    yR = paragraph(c,
        "Q(s,a) ← Q(s,a) + α [r + γ max<sub>a′</sub> Q(s′,a′) − Q(s,a)]",
        right_x, yR, COL_W, "equation", gap=2)
    q_tuning = metrics["q_learning_hyperparameter_tuning"]
    selected_q = q_tuning["selected_parameters"]
    yR = paragraph(c,
        f"Terminal transitions do not bootstrap future value. Each final agent trains for {metrics['training_episodes_per_seed']:,} episodes with α = {selected_q['learning_rate']:.2f}, γ = {selected_q['discount_factor']:.2f}, and ε decay = {selected_q['epsilon_decay']:.4f}; ε starts at 1.0 and floors at 0.05. A four-candidate, two-seed validation sweep selected α and decay before held-out testing. Five final training seeds assess robustness.",
        right_x, yR, COL_W, "body", gap=2)
    yR = paragraph(c,
        "Early in training, epsilon-greedy exploration tries different preparation batches; as epsilon decays, the agent increasingly uses actions with the highest current Q-value. Discounting values rewards that occur later in the simulated day, so a decision can account for carryover into the next slot.",
        right_x, yR, COL_W, "small", gap=2)

    yR = section(c, "9. Evaluation design", right_x, yR, COL_W)
    yR = paragraph(c,
        f"Each policy is tested on the same 2,000 held-out simulated days using matched seeds. The rule baseline receives the same forecast and history signals. Its target and history correction are tuned on {metrics['validation_tuning']['validation_episodes_per_candidate']:,} validation days; the selected history correction is {metrics['validation_tuning']['selected_history_adjustment']:.0%}, so the final rule does not adjust preparation decisions using sales history. Q-learning's learning rate and epsilon decay are tuned separately on a two-seed validation sweep. Five final training seeds, paired intervals, and demand-shift tests assess robustness.",
        right_x, yR, COL_W, "body", gap=2)
    yR = section(c, "10. Reproducibility", right_x, yR, COL_W)
    yR = paragraph(c,
        "The primary agent uses seed 2026; four more agents use independent seeds. Test episodes are separate from training and use matched demand draws across policies. Outputs include the Q-table, returns, metrics JSON, plots, policy slices, and an example trace. Twenty unit tests cover forecast inference, sales-history state encoding, transitions, rewards, configuration, training, and reproducibility.",
        right_x, yR, COL_W, "body", gap=2)
    yR = section(c, "11. Scope of evidence", right_x, yR, COL_W)
    yR = paragraph(c,
        "The baseline comparison supports conclusions only about this simulation and reward design. It does not establish that the same improvement would occur in a live canteen. Real deployment would require calibrated demand data, cost estimates, and operational and food-safety constraints.",
        right_x, yR, COL_W, "body", gap=2)

    if min(yL, yR) < 150:
        raise RuntimeError(f"Page 1 content leaves insufficient room for RL diagram: left y={yL:.1f}, right y={yR:.1f}")
    draw_rl_cycle(c, min(yL, yR) - 8)
    c.showPage()


def draw_page_two(c):
    draw_header(c, 2)
    y = PAGE_H - 112
    q_robust = metrics["multi_seed_robustness"]["q_learning_metrics"]["reward_mean"]
    y = paragraph(c,
        f"Main test: 2,000 held-out synthetic days per policy. Across five independently trained Q-learning agents, mean reward was {q_robust['mean_across_training_seeds']:.2f} +/- {q_robust['std_across_training_seeds']:.2f} SD; the table shows seed 2026 for paired test-day comparisons. These are simulated outcomes, not real-canteen measurements.",
        MARGIN, y, FULL_W, "body", gap=4)

    test_metrics = metrics["test_metrics"]
    q_key = "Q-learning (seed 2026)"
    fixed_key = "Fixed 15-portion batch"
    schedule_key = "Forecast-aware tuned rule"
    def fmt(key, metric):
        return f'{test_metrics[key][metric]:.2f}'
    data = [
        ["Policy", "Mean reward/day", "Sold/day", "Unmet demand/day", "Waste/day"],
        ["Q-learning (seed 2026)", fmt(q_key, "reward_mean"), fmt(q_key, "sold_mean"), fmt(q_key, "stockout_mean"), fmt(q_key, "waste_mean")],
        ["Fixed 15-portion batch", fmt(fixed_key, "reward_mean"), fmt(fixed_key, "sold_mean"), fmt(fixed_key, "stockout_mean"), fmt(fixed_key, "waste_mean")],
        ["Forecast-aware tuned rule", fmt(schedule_key, "reward_mean"), fmt(schedule_key, "sold_mean"), fmt(schedule_key, "stockout_mean"), fmt(schedule_key, "waste_mean")],
    ]
    col_ws = [145, 83, 63, 91, 74]
    y = draw_table(c, data, MARGIN, y, FULL_W, col_widths=col_ws, font=6.5)

    # Full-width comparison plot, generated by run_experiment.py.
    chart_path = RESULTS / "policy_comparison.png"
    img = ImageReader(str(chart_path))
    chart_w = FULL_W
    img_w, img_h = img.getSize()
    chart_h = chart_w * img_h / img_w
    y_img = y - chart_h - 2
    c.drawImage(img, MARGIN, y_img, width=chart_w, height=chart_h, preserveAspectRatio=True, mask="auto")
    y = y_img - 12
    y = paragraph(c, "Figure 1. Mean reward, waste, and unmet demand for the learned policy and comparison baselines.",
                  MARGIN, y + 5, FULL_W, "caption", gap=3)

    left_x = MARGIN
    right_x = MARGIN + COL_W + GUTTER
    col_top = y
    yL = col_top
    yR = col_top

    yL = section(c, "12. Learning behavior", left_x, yL, COL_W)
    duration = metrics.get("training_duration_validation", {})
    duration_summary = duration.get("summary", {})
    reward_6k = duration_summary.get("6000", {}).get("reward_mean", {}).get("mean_across_training_seeds", 24.10)
    reward_60k = duration_summary.get("60000", {}).get("reward_mean", {}).get("mean_across_training_seeds", 25.73)
    reward_delta = duration.get("paired_validation_reward_difference_60000_minus_6000", {}).get("mean_across_paired_training_seeds", 1.63)
    yL = paragraph(c,
        f"The curve plots 200-episode returns across five seeds (±1 SD). On 1,000 separate validation days per seed, 60,000 episodes scored {reward_60k:.2f} mean reward versus {reward_6k:.2f} for 6,000 episodes (+{reward_delta:.2f}/day across five matched training seeds). The 60,000-episode setting was selected before final held-out evaluation; gains are simulator-specific.",
        left_x, yL, COL_W, "small", gap=2)
    learning_path = RESULTS / "learning_curve.png"
    l_w = COL_W
    l_img = ImageReader(str(learning_path))
    l_img_w, l_img_h = l_img.getSize()
    l_h = l_w * l_img_h / l_img_w
    yL -= l_h
    c.drawImage(ImageReader(str(learning_path)), left_x, yL, width=l_w, height=l_h, preserveAspectRatio=True, mask="auto")
    yL -= 9
    yL = paragraph(c, "Figure 2. Mean return in 200-episode bins across five seeds; shaded band is ±1 SD.", left_x, yL, COL_W, "caption", gap=2)
    yL = section(c, "13. Interpretation and paired checks", left_x, yL, COL_W)
    primary_seed = str(metrics["robustness_training_seeds"][0])
    fixed_pair = metrics["paired_comparisons"][fixed_key]["reward"]["per_training_seed"][primary_seed]
    tuned_pair = metrics["paired_comparisons"][schedule_key]["reward"]["per_training_seed"][primary_seed]
    q_waste = test_metrics[q_key]["waste_mean"]
    baseline_waste = test_metrics[schedule_key]["waste_mean"]
    low_stress = metrics["demand_shift_stress_test"]["results"]["low_demand_0.75x"]
    high_stress = metrics["demand_shift_stress_test"]["results"]["high_demand_1.25x"]
    q_stress_key = "Q-learning (trained at nominal demand)"
    yL = paragraph(c,
        f"On the held-out test set, Q-learning reward was {test_metrics[q_key]['reward_mean']:.2f} (95% CI {test_metrics[q_key]['reward_ci95_low']:.2f} to {test_metrics[q_key]['reward_ci95_high']:.2f}) versus {test_metrics[schedule_key]['reward_mean']:.2f} for the tuned schedule (95% CI {test_metrics[schedule_key]['reward_ci95_low']:.2f} to {test_metrics[schedule_key]['reward_ci95_high']:.2f}). The paired reward difference (Q-learning minus baseline) was {tuned_pair['mean_difference']:.2f} (95% CI {tuned_pair['ci95_low']:.2f} to {tuned_pair['ci95_high']:.2f}); versus fixed-batch it was {fixed_pair['mean_difference']:.2f} ({fixed_pair['ci95_low']:.2f} to {fixed_pair['ci95_high']:.2f}). Q-learning discarded fewer portions ({q_waste:.2f} vs {baseline_waste:.2f}) but sold fewer and had more unmet demand than the tuned schedule. Across five training seeds, mean Q-learning reward was {metrics['multi_seed_robustness']['q_learning_metrics']['reward_mean']['mean_across_training_seeds']:.2f} (SD {metrics['multi_seed_robustness']['q_learning_metrics']['reward_mean']['std_across_training_seeds']:.2f}).",
        left_x, yL, COL_W, "small", gap=2)
    yL = paragraph(c,
        f"Under demand shifts, mean reward for Q-learning versus the nominal-planning tuned schedule was {low_stress[q_stress_key]['reward_mean']:.2f} vs {low_stress[schedule_key]['reward_mean']:.2f} at 0.75× demand, and {high_stress[q_stress_key]['reward_mean']:.2f} vs {high_stress[schedule_key]['reward_mean']:.2f} at 1.25×. Both policies keep their nominal planning assumptions; the results show sensitivity to a changed environment.",
        left_x, yL, COL_W, "tiny", gap=2)

    yR = section(c, "14. Limitations", right_x, yR, COL_W)
    yR = paragraph(c,
        "Demand is synthetic and Poisson-distributed. The environment assumes one meal type, eight time slots, a forecast with assumed 70% accuracy, a 20-portion maximum action, simplified carryover, and illustrative reward weights. Sales pace and prior stockouts summarize history but do not maintain a full belief over the persistent hidden regime. No real-world savings are established.",
        right_x, yR, COL_W, "small", gap=3)
    yR = section(c, "15. Reward-weight sensitivity", right_x, yR, COL_W)
    sens_rows = [["Train waste wt.", "Base reward/day", "Waste/day", "Unmet/day"]]
    for row in sensitivity["results"]:
        means = row["mean_metrics_under_common_base_reward"]
        sds = row["sd_of_seed_level_metrics"]
        sens_rows.append([
            f"{row['training_waste_penalty']:.2f}",
            f"{means['reward_mean']:.2f} +/- {sds['reward_mean']:.2f}",
            f"{means['waste_mean']:.2f} +/- {sds['waste_mean']:.2f}",
            f"{means['stockout_mean']:.2f} +/- {sds['stockout_mean']:.2f}",
        ])
    yR = draw_table(c, sens_rows, right_x, yR, COL_W, col_widths=[49, 78, 62, 65], font=5.3)
    yR = paragraph(c,
        "Each setting trains three seeds; each policy is tested on the same 2,000 held-out scenarios using the original reward weights. Values are mean +/- SD across trained seeds, not confidence intervals. As the training waste penalty rises, waste falls but unmet demand rises. This demonstrates reward sensitivity, not an objectively optimal coefficient.",
        right_x, yR, COL_W, "tiny", gap=3)
    yR = section(c, "16. Future work", right_x, yR, COL_W)
    yR = paragraph(c,
        "Future work could maintain a richer belief state, test different forecast accuracies, calibrate assumptions with permitted anonymized records, and model freshness, lead times, and food-safety constraints.",
        right_x, yR, COL_W, "small", gap=3)
    yR = section(c, "Conclusion", right_x, yR, COL_W)
    yR = paragraph(c,
        "Within this noisy-forecast simulation, Q-learning is compared with a fixed batch and a validation-tuned forecast-aware rule using the same observations. The sales-pace and prior-stockout features provide a compact summary of past outcomes but not a full belief state. Results are simulator-specific and show trade-offs rather than a universal RL win.",
        right_x, yR, COL_W, "small", gap=3)
    yR = paragraph(c,
        "Reproducibility: source, trained Q-table, tests, metrics, sensitivity analysis, and figure scripts are included in this repository.",
        right_x, yR, COL_W, "tiny", gap=2)

    if min(yL, yR) < 37:
        raise RuntimeError(f"Page 2 content overflow: left y={yL:.1f}, right y={yR:.1f}")
    c.showPage()


def main():
    c = canvas.Canvas(str(OUT), pagesize=A4)
    c.setTitle("Adaptive Meal Preparation in a Campus Canteen using Q-Learning")
    c.setAuthor("AI Assignment 2 Project Team")
    draw_page_one(c)
    draw_page_two(c)
    c.save()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
