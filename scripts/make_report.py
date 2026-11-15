"""Build docs/report.pdf, the short risk note, from the package and the frozen data.

    pip install -e ".[dev]"      # reportlab is in the dev extras
    python scripts/make_report.py

Every number in the text is computed here, so the note always matches the code.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from adaptive_ltv import environment as env
from adaptive_ltv.analysis import default_controller
from adaptive_ltv.inputs import DATA, load_case_shiller, utilization_signal
from adaptive_ltv.plots import plot_decision
from adaptive_ltv.scenarios import market_shock, merton_market_scenarios, operational_collapse

DOCS = DATA.parent / "docs"
FIG = DOCS / "report_figures"
OUT = DOCS / "report.pdf"
REPO = "github.com/ChristopheAbelem/adaptive-ltv-rwa"
APP = "adaptive-ltv-rwa-hunbgjeqcfmyytwhpgbdzv.streamlit.app"
AS_OF = "10 November 2026"

NAVY = colors.HexColor("#1f3b5c")
GREY_TXT = colors.HexColor("#5f6b7a")
ROW = colors.HexColor("#eef2f7")
FIXED, FUZZY, ACCENT = "#9a9a9a", "#2a6fdb", "#c0392b"

# ------------------------------------------------------------------------------------------------ fonts and styles
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"      # ships with matplotlib: portable
pdfmetrics.registerFont(TTFont("DV", FONT_DIR / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", FONT_DIR / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVI", FONT_DIR / "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

S = {
    "title": ParagraphStyle("title", fontName="DVB", fontSize=20, leading=24, textColor=NAVY, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=11, leading=14, textColor=GREY_TXT, spaceAfter=4),
    "by": ParagraphStyle("by", fontName="DV", fontSize=7.5, leading=10, textColor=GREY_TXT, spaceAfter=8),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=13, leading=17, textColor=NAVY, spaceBefore=4, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=10, leading=13, textColor=NAVY, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="DV", fontSize=8.6, leading=11.6, spaceAfter=5, alignment=TA_LEFT),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=8.6, leading=11.4, leftIndent=9, bulletIndent=0,
                             spaceAfter=4),
    "cap": ParagraphStyle("cap", fontName="DV", fontSize=7.2, leading=9.2, textColor=GREY_TXT, spaceAfter=7),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.6, leading=9.4),
    "cellb": ParagraphStyle("cellb", fontName="DVB", fontSize=7.6, leading=9.4),
    "src": ParagraphStyle("src", fontName="DV", fontSize=7, leading=9, textColor=GREY_TXT),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header=True, align_right_from=1, bold_last=False):
    data = [[P(c, "cellb" if (header and i == 0) or (bold_last and i == len(rows) - 1) else "cell")
             if isinstance(c, str) else c for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths)
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2), ("LEFTPADDING", (0, 0), (-1, -1), 3),
             ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY)]
    for i in range(1 if header else 0, len(rows)):
        if (i - (1 if header else 0)) % 2 == 1:
            style.append(("BACKGROUND", (0, i), (-1, i), ROW))
    style.append(("LINEBELOW", (0, len(rows) - 1), (-1, len(rows) - 1), 0.4, colors.HexColor("#c8d0da")))
    t.setStyle(TableStyle(style))
    return t


def figure(path, width_mm, caption):
    img = Image(str(path))
    w = width_mm * mm
    img.drawHeight = img.drawHeight * w / img.drawWidth
    img.drawWidth = w
    return KeepTogether([img, Spacer(1, 2), P(caption, "cap")])


def frame(canvas, doc):
    w, h = A4
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 13 * mm, w, 13 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("DVB", 8)
    canvas.drawString(18 * mm, h - 8.3 * mm, "REALT LENDING MARKET (RMM, GNOSIS CHAIN)  |  RISK PARAMETER NOTE")
    canvas.setFont("DV", 8)
    canvas.drawRightString(w - 18 * mm, h - 8.3 * mm, f"Data as of {AS_OF}")
    canvas.setFillColor(GREY_TXT)
    canvas.setFont("DV", 6.8)
    canvas.drawString(18 * mm, 10 * mm, "Academic portfolio project. Not investment advice. Christophe Abelem, "
                                        "github.com/ChristopheAbelem")
    canvas.drawRightString(w - 18 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


# ------------------------------------------------------------------------------------------------ numbers
def compute():
    ctrl = default_controller()
    collapse, shock = operational_collapse(), market_shock()
    n = {"ctrl": ctrl, "collapse": collapse, "shock": shock}
    for key, sc in [("col", collapse), ("sh", shock)]:
        n[f"{key}_fixed"] = env.run(sc, env.fixed_policy())
        n[f"{key}_fuzzy"] = env.run(sc, ctrl)
        n[f"{key}_sum"] = {name: env.loss_summary(n[f"{key}_{name}"]) for name in ("fixed", "fuzzy")}
    n["seeds"] = {name: np.mean([env.loss_summary(env.run(collapse, pol, seed=k))["total_loss_%"] for k in range(20)])
                  for name, pol in [("fixed", env.fixed_policy()), ("fuzzy", ctrl)]}

    def loss(sc, pol, key=None, value=None):
        if key is None:
            return env.loss_summary(env.run(sc, pol))["total_loss_%"]
        with env.assumption(key, value):
            return env.loss_summary(env.run(sc, pol))["total_loss_%"]

    checks = [("Base case", None, None),
              ("Repay 0% of the excess a month", "deleverage_rate", 0.0),
              ("Repay 5% a month", "deleverage_rate", 0.05),
              ("Repay 15% a month", "deleverage_rate", 0.15),
              ("Token discount without rent 40%", "token_discount", 0.4),
              ("Token discount without rent 80%", "token_discount", 0.8),
              ("No forced-sale price impact", "price_impact", 0.0),
              ("Price impact 0.75", "price_impact", 0.75),
              ("Appraisal every 6 months", "appraisal_months", 6),
              ("Appraisal every 24 months", "appraisal_months", 24)]
    n["robust"] = [(label, loss(collapse, env.fixed_policy(), k, v), loss(collapse, ctrl, k, v))
                   for label, k, v in checks]
    n["robust_shock"] = [(label, loss(shock, env.fixed_policy(), k, v), loss(shock, ctrl, k, v))
                         for label, k, v in checks[:4] + checks[6:9]]

    _, paths = merton_market_scenarios(n_paths=300, months=84, seed=0)
    mc = {name: np.array([env.loss_summary(env.run(sc, pol))["total_loss_%"] for sc in paths])
          for name, pol in [("fixed", env.fixed_policy()), ("fuzzy", ctrl)]}
    n["mc"] = {name: {"share_gt1": (v > 1).mean(), "worst": v.max(), "p99": np.percentile(v, 99)}
               for name, v in mc.items()}

    # the decision taken with July 2025 signals in the closed loop
    trace = {}

    def recording(v, u, r):
        trace[len(trace)] = ctrl.decide(v, u, r)
        return trace[len(trace) - 1].max_ltv

    env.run(collapse, recording)
    n["jul25"] = trace[list(collapse.house_index.index).index(pd.Timestamp("2025-07-01"))]
    floor = n["col_fuzzy"]["max_ltv"]
    n["floor_month"] = floor[floor <= 0.3001].index[0]

    # real-world series
    liq = pd.read_csv(DATA / "raw" / "rmm_liquidations.csv", parse_dates=["date"])
    n["liq_m"] = liq.set_index("date").resample("MS")["tx"].count()
    pool = pd.read_csv(DATA / "raw" / "rmm_liquidity_daily.csv", parse_dates=["date"], index_col="date")
    n["pool_m"] = pool.resample("MS").mean()
    n["pool_last"] = pool.iloc[-1]
    n["util_real"] = utilization_signal()
    n["cs"] = load_case_shiller()
    return n


# ------------------------------------------------------------------------------------------------ figures
def style_axes(ax):
    ax.grid(alpha=0.3, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def pct(ax, decimals=0):
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.{decimals}%}"))


def dates(ax, yearly):
    loc = mdates.YearLocator() if yearly else mdates.MonthLocator(bymonth=[1, 7])
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def fig_closed_loop(fixed, fuzzy, title, path, yearly):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for r, name, color, ls in [(fixed, "fixed 60%", FIXED, "--"), (fuzzy, "fuzzy controller", FUZZY, "-")]:
        axes[0].plot(r.index, r["max_ltv"], color=color, ls=ls, lw=2, label=name)
        axes[1].plot(r.index, r["utilization"], color=color, ls=ls, lw=2)
        axes[2].plot(r.index, r["loss_to_date"] / r.attrs["initial_supply"], color=color, ls=ls, lw=2)
    for ax, t in zip(axes, ["Max LTV (control action)", "Pool utilization", "Lenders' loss to date, % of deposits"]):
        ax.set_title(t, loc="left", fontsize=9.5)
        style_axes(ax)
        dates(ax, yearly)
    pct(axes[0]); pct(axes[1])
    worst = max((fixed["loss_to_date"] / fixed.attrs["initial_supply"]).max(), 0.001)
    pct(axes[2], 1 if worst < 0.1 else 0)
    axes[0].set_ylim(0.25, 0.65)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_real_world(n, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    start, end = "2024-02-01", "2026-10-31"
    pm = n["pool_m"].loc[start:end]
    axes[0].plot(pm.index, pm["borrowed_stable"] / 1e6, color=FUZZY, lw=2, label="lent to borrowers")
    axes[0].plot(pm.index, pm["tvl_stable"] / 1e6, color=FIXED, lw=2, label="idle cash lenders can withdraw")
    axes[0].set_title("RMM stablecoins, $ mn (DefiLlama)", loc="left", fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=7.5, loc="center left")
    lm = n["liq_m"].loc[start:end]
    axes[1].bar(lm.index, lm.values, width=22, color=ACCENT)
    axes[1].set_title("Liquidations per month (on-chain)", loc="left", fontsize=9.5)
    cs = n["cs"].loc[start:]
    axes[2].plot(cs.index, cs / cs.iloc[0] - 1, color="#333333", lw=2)
    pct(axes[2])
    axes[2].set_title("Detroit house prices since Feb 2024", loc="left", fontsize=9.5)
    for ax in axes:
        style_axes(ax)
        dates(ax, False)
        ax.axvline(pd.Timestamp("2025-07-01"), color="#555555", lw=0.8, ls=":")
        ax.set_xlim(pd.Timestamp(start), pd.Timestamp(end))
    axes[1].text(pd.Timestamp("2025-07-10"), axes[1].get_ylim()[1] * 0.92, "rent escrow", fontsize=7.5,
                 color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_decision(n, path):
    plt.rcParams.update({"font.size": 8})
    fig = plot_decision(n["ctrl"], n["jul25"])
    fig.set_size_inches(12, 3.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def fig_robustness(n, path):
    rows = n["robust"]
    labels = [r[0] for r in rows][::-1]
    fx = [r[1] for r in rows][::-1]
    fz = [r[2] for r in rows][::-1]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.barh(y + 0.2, fx, height=0.38, color=FIXED, label="fixed 60%")
    ax.barh(y - 0.2, fz, height=0.38, color=FUZZY, label="fuzzy controller")
    for yi, a, b in zip(y, fx, fz):
        ax.text(a + 0.8, yi + 0.2, f"{a:.1f}", va="center", fontsize=7, color="#555555")
        ax.text(b + 0.8, yi - 0.2, f"{b:.1f}", va="center", fontsize=7, color=FUZZY)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("lenders' loss, % of deposits", fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ document
def build(n):
    FIG.mkdir(parents=True, exist_ok=True)
    fig_closed_loop(n["col_fixed"], n["col_fuzzy"], "Operational collapse (RealT, Feb 2024 - Jul 2026), simulated",
                    FIG / "f1_collapse.png", yearly=False)
    fig_real_world(n, FIG / "f2_real.png")
    fig_decision(n, FIG / "f3_decision.png")
    fig_closed_loop(n["sh_fixed"], n["sh_fuzzy"], "Market shock (Detroit house prices, 2006 - 2012), simulated",
                    FIG / "f4_shock.png", yearly=True)
    fig_robustness(n, FIG / "f5_robust.png")

    cf, cz = n["col_sum"]["fixed"], n["col_sum"]["fuzzy"]
    sf, sz = n["sh_sum"]["fixed"], n["sh_sum"]["fuzzy"]
    rb = {r[0]: r[1:] for r in n["robust"]}
    rs = {r[0]: r[1:] for r in n["robust_shock"]}
    d = n["jul25"]
    liq = n["liq_m"]
    liq_crisis = int(liq.loc["2025-07":"2026-10"].sum())
    liq_before = int(liq.loc[:"2025-06"].sum())
    pm = n["pool_m"]
    lent_peak = pm["borrowed_stable"].max() / 1e6
    lent_last = n["pool_last"]["borrowed_stable"] / 1e6
    idle_dec = pm.loc["2025-12", "tvl_stable"].iloc[0] / 1e6
    cs = n["cs"]
    cs_change = cs.loc["2026-07-01"] / cs.loc["2024-02-01"] - 1
    cs_peak = cs.idxmax()
    floor_m = n["floor_month"].strftime("%B %Y")
    mcf, mcz = n["mc"]["fixed"], n["mc"]["fuzzy"]
    jul_mem = {k: {t: x for t, x in v.items() if x > 0} for k, v in d.memberships.items()}
    n_rules = sum(s > 0 for _, s in d.activations)

    story = []
    story += [P("Adaptive LTV for Tokenized Real Estate: What RealT's Collapse Says About Fixed Risk "
                "Parameters", "title"),
              P("A fuzzy closed-loop controller for the max loan-to-value, tested on a simulation of RealT's "
                "lending market", "sub"),
              P(f"Christophe Abelem, Computer Engineering, CESUPA (Belém, Brazil) | Model and data: {REPO} | "
                f"Live app: {APP}", "by")]

    key = table([
        ["Key data", ""],
        ["RMM parameters for RealToken collateral (on-chain): max LTV / liquidation / bonus", "60% / 70% / 10%"],
        ["Stablecoins lent: peak (Dec 2025) / latest", f"${lent_peak:.1f} mn / ${lent_last:.1f} mn"],
        ["Idle cash for lenders: Dec 2025 / since Jan 2026", f"${idle_dec:.1f} mn / $0"],
        ["Liquidations: Jun 2024 - Jun 2025 / Jul 2025 - Oct 2026", f"{liq_before} / {liq_crisis:,}"],
        ["Detroit house prices, Feb 2024 to Jul 2026", f"{cs_change:+.0%}"],
        ["Simulated lenders' loss: fixed 60% / fuzzy", f"{cf['total_loss_%']:.1f}% / {cz['total_loss_%']:.1f}%"],
        ["Same, mean of 20 loan books", f"{n['seeds']['fixed']:.1f}% / {n['seeds']['fuzzy']:.1f}%"],
        ["Same, borrowers repay nothing", f"{rb['Repay 0% of the excess a month'][0]:.1f}% / "
                                          f"{rb['Repay 0% of the excess a month'][1]:.1f}%"],
    ], [62 * mm, 26 * mm])
    summary = [P("Summary", "h2")] + bullets([
        "<b>A fixed 60% limit could not see RealT's crisis coming.</b> Rent stopped from July 2025 and the pool "
        "ran out of cash in January 2026, while Detroit house prices kept rising: no price signal existed.",
        f"<b>A controller that reads rent and liquidity cuts simulated losses from {cf['total_loss_%']:.0f}% to "
        f"{cz['total_loss_%']:.0f}% of deposits.</b> It reaches its 30% floor in {floor_m}, two months before the "
        "pool empties.",
        "<b>The gain requires existing loans to shrink.</b> If borrowers repay nothing when the limit falls, "
        "both rules lose the same.",
        f"<b>Against a pure price crash it adds little:</b> a fixed 60% loses {sf['total_loss_%']:.1f}% in the "
        "Detroit 2006-2012 replay, and the comparison flips with the assumptions.",
        "<b>Recommendation:</b> lending markets on income-producing collateral should tie their limits to "
        "income and liquidity, and apply cuts to existing positions gradually.",
    ])
    top = Table([[key, summary]], colWidths=[92 * mm, 82 * mm])
    top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("RIGHTPADDING", (0, 0), (0, 0), 6)]))
    story += [top, Spacer(1, 6),
              figure(FIG / "f1_collapse.png", 174,
                     "Figure 1. Simulated replay of RealT's lending market under its fixed 60% max LTV (grey) and "
                     "under the fuzzy controller (blue): the limit chosen each month, the share of the pool lent "
                     "out, and the lenders' loss to date (realized bad debt plus debt the collateral no longer "
                     "covers)."),
              P("<b>Question and approach.</b> Could a lending pool backed by tokenized houses have avoided most of "
                "RealT's losses by adjusting its maximum loan-to-value every month instead of keeping it fixed? A "
                "Mamdani fuzzy controller reads three monthly signals (price volatility, pool utilization, rent "
                "coverage) and sets the limit between 30% and 60%. It is tested in a closed-loop simulation of the "
                "pool, where each decision changes the signals of the next month, and compared with the fixed rule "
                "in RealT's collapse, in Detroit's 2006-2012 crash and in 300 random price futures.")]

    # ---- 1. case
    story += [PageBreak(), P("1. What happened at RealT", "h1"),
              P("RealT sold fractions of about 700 rented Detroit houses as tokens that pay a share of the rent. In "
                "its lending market (RMM) holders could deposit the tokens and borrow stablecoins, up to 60% of the "
                "collateral's appraised value, with liquidation at 70%. The appraisal is updated only "
                "periodically, so the pool sees a slow, smoothed price."),
              table([["Date", "Event", "Source"],
                     ["Jul 2025", "Rent redirected to escrow by court order", "Press (Outlier Media, BridgeDetroit)"],
                     ["Jul - Dec 2025", "Liquidations rise from a few a month to 30-160 a month", "RMM contracts"],
                     ["Late 2025", "Rent distributions largely halted", "Press"],
                     ["Jan 2026", f"Idle cash reaches zero (from ${idle_dec:.1f} mn); 269 liquidations", "DefiLlama, RMM"],
                     ["Feb 2026", "Rent formally suspended", "Press (Lofty)"],
                     ["Jul 2026", "RealT announces its liquidation", "Press"],
                     ["Sep 2026", f"{int(liq.loc['2026-09'].sum()):,} liquidations in one month", "RMM contracts"]],
                    [24 * mm, 100 * mm, 50 * mm]),
              Spacer(1, 6),
              figure(FIG / "f2_real.png", 174,
                     "Figure 2. Observed data. Left: stablecoins lent and idle cash in the RMM pool, monthly mean. "
                     "Centre: liquidation events read from the RMM contracts. Right: S&amp;P Cotality Case-Shiller "
                     "MI-Detroit index (FRED DEXRNSA). Dotted line: start of the rent escrow."),
              P(f"<b>What the data shows.</b> The crisis was operational, not a price crash. Detroit prices rose "
                f"{cs_change:.0%} between February 2024 and July 2026 and peaked in {cs_peak:%B %Y}, so any rule "
                "driven by collateral prices would have stayed calm. What changed was income and liquidity: "
                "liquidations started climbing in the month the rent escrow began, and the pool's idle cash fell to "
                f"zero in January 2026. Borrowed stablecoins have stayed near ${lent_last:.0f} mn since then: "
                "lenders cannot withdraw, and the tokens backing the loans no longer earn rent."),
              P("<b>Why the fixed limit failed.</b> A token's value comes from its rent. When rent stops, buyers pay "
                "far less than the house is worth, but the appraisal the pool uses does not move until the next "
                "valuation. Loans that look safe at 60% of the appraisal are no longer covered by what the "
                "collateral can actually be sold for.")]

    # ---- 2. model
    story += [PageBreak(), P("2. The controller and the simulated pool", "h1"),
              P("<b>Controller.</b> Mamdani fuzzy inference with three inputs and one output (the max LTV, 30-60%). "
                "Each input is described by three linguistic terms; the 27 rules combine one term of each input and "
                "were built from severity scores (volatility 0-2, utilization 0-4, rent 0-4), so that a liquidity "
                "crunch or failed rent alone sets the 30% floor while price volatility alone lowers the limit to "
                "45%. A calm month gives exactly 60%, the same as RealT's rule. Mamdani was chosen over "
                "Takagi-Sugeno because every rule stays a sentence in words and every step can be shown."),
              table([["Input", "Source", "Terms"],
                     ["Volatility of the collateral price", "Case-Shiller Detroit, seasonally adjusted, de-smoothed "
                      "(Geltner, rho 0.725), 12-month rolling; in the simulation, the token's market price",
                      "low / medium / high (historical percentiles)"],
                     ["Pool utilization (lent / deposited)", "DefiLlama, RMM; in the simulation, the pool's state",
                      "healthy / tight / crunch"],
                     ["Rent coverage (share of rent paid)", "Stylized path between the documented dates above",
                      "full / partial / low"]],
                    [44 * mm, 86 * mm, 44 * mm]),
              Spacer(1, 6),
              figure(FIG / "f3_decision.png", 174,
                     f"Figure 3. The decision taken with July 2025 signals in the simulated collapse. Each input sits "
                     f"between two terms (dots: degrees of membership), so {n_rules} of the 27 rules fire; each "
                     f"cuts its output triangle at its strength (dashed lines), the cuts are joined, and the centroid "
                     f"({d.max_ltv:.1%}) becomes the limit for August."),
              P(f"<b>Reading Figure 3.</b> Volatility {d.inputs['volatility']:.1%} is "
                f"{jul_mem['volatility'].get('medium', 0):.2f} medium; utilization {d.inputs['utilization']:.1%} is "
                f"{jul_mem['utilization'].get('tight', 0):.2f} tight; rent coverage {d.inputs['rent_coverage']:.0%} is "
                f"{jul_mem['rent_coverage'].get('partial', 0):.2f} partial. None of the three is in its worst "
                f"term, but three moderate warnings together already lower the limit by "
                f"{60 - 100 * d.max_ltv:.0f} points, gradually rather than at a threshold."),
              P("<b>Simulated pool.</b> One month at a time: 200 borrowers adjust their loans to the limit and "
                "repay 10% of any excess a month; interest is paid out of rent and unpaid interest is added to the "
                "debt; lenders run (up to 17% of deposits a month, the largest fall observed) once rent coverage "
                "drops to 10%, limited by idle cash; positions above 70% of a yearly appraisal are liquidated at "
                "the token's market price, and forced sales push that price down. The token is worth the house "
                "value minus a 60% discount when rent stops. The loop is closed: the limit changes utilization and "
                "the market price, which are the next month's inputs."),
              P("<b>Validation.</b> 45 automated tests (de-smoothing, coverage of every fuzzy variable, all 27 "
                "rules, monotonicity, accounting identities of the pool) run on every commit. The simulated "
                "utilization tracks the real pool within 2.5 points on average. The Merton jump-diffusion used for "
                "random futures reproduces its closed-form growth and variance within 0.2%.")]

    # ---- 3. results
    story += [PageBreak(), P("3. Results", "h1"),
              P(f"<b>Operational collapse.</b> The controller keeps 60% until early 2025, lowers the limit as "
                f"utilization and then rent deteriorate, and reaches 30% in {floor_m}. Borrowers shrink their loans "
                "in the meantime, so when the pool empties in January 2026 no position crosses the liquidation "
                "threshold. Under the fixed rule, loans stay large, unpaid interest and the empty pool push them "
                "over the threshold, and the collateral is sold far below its appraisal (Figure 1)."),
              table([["Operational collapse", "Fixed 60%", "Fuzzy"],
                     ["Liquidations", f"{cf['liquidations']:.0f}", f"{cz['liquidations']:.0f}"],
                     ["Realized bad debt, % of deposits", f"{cf['realized_bad_debt_%']:.1f}%",
                      f"{cz['realized_bad_debt_%']:.1f}%"],
                     ["Debt not covered by collateral at the end", f"{cf['uncovered_debt_at_end_%']:.1f}%",
                      f"{cz['uncovered_debt_at_end_%']:.1f}%"],
                     ["Total loss, % of deposits", f"{cf['total_loss_%']:.1f}%", f"{cz['total_loss_%']:.1f}%"],
                     ["Total loss, mean of 20 loan books", f"{n['seeds']['fixed']:.1f}%",
                      f"{n['seeds']['fuzzy']:.1f}%"]],
                    [90 * mm, 30 * mm, 30 * mm], bold_last=False),
              Spacer(1, 6),
              figure(FIG / "f4_shock.png", 174,
                     "Figure 4. Simulated replay of Detroit's 2006-2012 house-price crash with rent paid throughout."),
              P(f"<b>Market shock.</b> House prices fall by half but rent keeps being paid. The controller lowers the "
                f"limit to 45% as volatility rises from mid-2007 and avoids the fixed rule's "
                f"{sf['total_loss_%']:.1f}% loss in the base case. The result is not robust: with no forced-sale "
                f"price impact neither rule loses anything, and with an appraisal every 6 months or faster "
                f"repayment the controller loses more ({rs['Appraisal every 6 months'][1]:.1f}% and "
                f"{rs['Repay 15% a month'][1]:.1f}% against {rs['Appraisal every 6 months'][0]:.1f}% and "
                f"{rs['Repay 15% a month'][0]:.1f}%). With an appraisal that moves in steps, losses depend on "
                "whether liquidations arrive spread out or all at once, and a lower limit can make that worse."),
              P(f"<b>Random futures.</b> Over 300 simulated 7-year Detroit price paths with rent paid throughout, "
                f"the fixed rule loses more than 1% of deposits in {mcf['share_gt1']:.0%} of them (worst "
                f"{mcf['worst']:.1f}%), the controller in {mcz['share_gt1']:.0%} (worst {mcz['worst']:.2f}%). Price "
                "risk alone rarely costs a 60% limit much.")]

    # ---- 4. robustness and recommendations
    story += [PageBreak(), P("4. How robust is the result?", "h1"),
              figure(FIG / "f5_robust.png", 140,
                     "Figure 5. Lenders' loss in the operational collapse when one assumption at a time changes."),
              P(f"<b>Reading.</b> The direction holds in every case, and the controller's loss barely moves with "
                f"the price impact or the appraisal frequency ({min(r[2] for r in n['robust'][6:]):.1f}% to "
                f"{max(r[2] for r in n['robust'][6:]):.1f}%). Two behaviours without public data decide the size: "
                f"<b>how fast borrowers repay</b> when the limit falls (0%: no benefit; 5%: "
                f"{rb['Repay 5% a month'][1]:.1f}%; 15%: {rb['Repay 15% a month'][1]:.1f}%) and <b>how much a "
                f"token loses</b> when rent stops (a 40% discount: {rb['Token discount without rent 40%'][0]:.1f}% "
                f"against {rb['Token discount without rent 40%'][1]:.1f}%; 80%: "
                f"{rb['Token discount without rent 80%'][0]:.1f}% against "
                f"{rb['Token discount without rent 80%'][1]:.1f}%)."),
              P("5. Recommendations for lending markets on real-world assets", "h1"),
              P("What the simulation suggests for protocols that accept income-producing tokens as collateral:"),
              *bullets([
                  "<b>Read income, not only price.</b> Rent coverage (or the equivalent cash flow of the asset) "
                  "should be a risk input. In RealT's case it was the first signal to move, and in the simulation "
                  "the limit starts falling as soon as coverage drops below 95%.",
                  "<b>Read the pool's own liquidity.</b> Utilization above about 86-90% should tighten the limit "
                  "before lenders lose access to their cash, not after.",
                  "<b>Make limit cuts reach existing loans.</b> Most lending protocols apply a lower LTV only to new "
                  "borrowing. Without a rule that makes positions shrink gradually (10% of the excess a month in "
                  "the base case), lowering the limit protects nobody.",
                  "<b>Mark collateral to what it can be sold for.</b> In the simulation, an appraisal refreshed once a "
                  "year hid a token that had lost most of its value; when income stops, the oracle price should be cut at once.",
                  "<b>Keep the rules readable.</b> Fuzzy rules let a risk team state its policy in words and still "
                  "obtain a gradual, auditable limit, instead of jumps at fixed thresholds.",
              ]),
              P("6. Limitations", "h1"),
              *bullets([
                  "<b>Stylized rent.</b> No public rent series exists; its timing comes from documented dates. The "
                  "note shows how the controller reacts when rent falls, not that rent data would have warned in "
                  "time.",
                  "<b>Behavioural assumptions.</b> Repayment speed, token discount and price impact have no public "
                  "data; Figure 5 shows how much they matter.",
                  "<b>Overlapping signals.</b> In the collapse the token's price falls because rent falls, so "
                  "volatility partly repeats the rent signal (reading the data volatility instead gives 6.1%).",
                  "<b>Simplifications.</b> One pool, monthly steps, identical collateral units; real pools react "
                  "daily and hold many different properties.",
              ]),
              Spacer(1, 4),
              P("<b>Sources.</b> RMM v3 contracts on Gnosis Chain (risk parameters, liquidation events, read via "
                "public RPC); DefiLlama (realt-rmm-marketplace-v2); FRED DEXRNSA, S&amp;P Cotality Case-Shiller "
                "MI-Detroit Home Price Index; LlamaRisk RMM review (June 2025); Outlier Media, BridgeDetroit and "
                "Lofty reporting on the RealT rent escrow and suspension; Geltner (1993) on appraisal smoothing; "
                f"Merton (1976) on jump-diffusion. Full code, data and tests: {REPO}.", "src")]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm,
                            bottomMargin=16 * mm, title="Adaptive LTV for Tokenized Real Estate: What RealT's "
                            "Collapse Says About Fixed Risk Parameters", author="Christophe Abelem",
                            subject="Risk parameter note: fuzzy closed-loop max LTV for tokenized real estate")
    doc.build(story, onFirstPage=frame, onLaterPages=frame)
    print(OUT)


if __name__ == "__main__":
    build(compute())
