"""Real-time simulation of the adaptive LTV controller.

Run from the repository root:  streamlit run app.py

Two tabs:
- Simulation: the lending pool runs month by month under two rules at once (RealT's fixed 60% and the fuzzy
  controller); the charts and the controller's internal steps update every month.
- Try the controller: move the three inputs by hand and see each Mamdani step.
"""

import time

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from adaptive_ltv import environment as env
from adaptive_ltv.analysis import active_rules, default_controller
from adaptive_ltv.plots import STYLE, plot_decision
from adaptive_ltv.scenarios import market_shock, operational_collapse

matplotlib.use("Agg")
plt.rcParams.update(STYLE)

TEXT = {
    "en": {
        "title": "Adaptive LTV: a fuzzy controller for loans backed by tokenized houses",
        "intro": (
            "A lending pool lends money to people who pledge **tokens** (digital shares of a rented house) as "
            "guarantee. The **max LTV** is the largest loan allowed per dollar of guarantee: at 60%, a token worth "
            "\\$100 lets you borrow up to \\$60. RealT keeps it fixed at 60%. The **fuzzy controller** reads three "
            "signals every month (how much prices swing, how much of the pool is already lent, how much rent is "
            "being paid) and lowers the limit between 60% and 30% when they get worse. The loop is closed: the new "
            "limit changes the pool, and the pool feeds the next month's signals."),
        "tab_sim": "Simulation (closed loop, real time)", "tab_try": "Try the controller",
        "scenario": "Scenario", "speed": "Seconds per simulated month",
        "delev": "Share of the excess loan repaid per month when the limit falls",
        "discount": "Token discount when rent stops",
        "assumptions": "Assumptions (sensitivity)", "start": "Start simulation",
        "month": "Month", "fuzzy_ltv": "Fuzzy max LTV", "loss_fixed": "Loss, fixed 60%", "loss_fuzzy": "Loss, fuzzy",
        "loss_help": "Losses of the lenders (unpaid debt) as % of the initial deposits",
        "ch_ltv": "Max LTV", "ch_util": "Pool utilization (lent / deposited)", "ch_signals": "Signals (fuzzy run)",
        "ch_loss": "Lenders' loss to date, % of deposits",
        "inside": "Inside the controller this month",
        "inside_help": ("1-3: each input becomes degrees of membership (0 to 1) in its terms. Rules combine one term "
                        "of each input with AND (the smallest degree). 4: each fired rule cuts its output triangle "
                        "at its strength, the cuts are joined (max), and the centre of gravity of the shape "
                        "(centroid) is the new limit."),
        "rules": "Rules fired (strength = smallest of the three degrees)",
        "done": "Simulation finished.",
        "vol": "Volatility (annual)", "util": "Utilization", "rent": "Rent coverage",
        "try_intro": "Move the inputs: the four steps of the controller are recomputed at once.",
        "result": "Max LTV",
        "scenarios": {"collapse": "Operational collapse (RealT 2024-2026): rent stops, lenders leave",
                      "shock": "Market shock (Detroit 2006-2012): house prices fall by half"},
    },
    "pt": {
        "title": "LTV adaptativo: controlador fuzzy para empréstimos garantidos por casas tokenizadas",
        "intro": (
            "Um pool empresta dinheiro a quem deixa **tokens** (frações digitais de uma casa alugada) como "
            "garantia. O **LTV máximo** é o maior empréstimo permitido por dólar de garantia: com 60%, um token de "
            "\\$100 permite pegar até \\$60. A RealT mantém esse limite fixo em 60%. O **controlador fuzzy** lê "
            "três sinais todo mês (quanto os preços oscilam, quanto do pool já está emprestado, quanto do aluguel "
            "está sendo pago) e reduz o limite entre 60% e 30% quando eles pioram. A malha é fechada: o novo "
            "limite muda o pool, e o pool alimenta os sinais do mês seguinte."),
        "tab_sim": "Simulação (malha fechada, tempo real)", "tab_try": "Testar o controlador",
        "scenario": "Cenário", "speed": "Segundos por mês simulado",
        "delev": "Parte do excesso de dívida devolvida por mês quando o limite cai",
        "discount": "Desconto do token quando o aluguel para",
        "assumptions": "Hipóteses (sensibilidade)", "start": "Iniciar simulação",
        "month": "Mês", "fuzzy_ltv": "LTV máximo fuzzy", "loss_fixed": "Perda, fixo 60%", "loss_fuzzy": "Perda, fuzzy",
        "loss_help": "Perda dos credores (dívida não paga) em % dos depósitos iniciais",
        "ch_ltv": "LTV máximo", "ch_util": "Utilização do pool (emprestado / depositado)",
        "ch_signals": "Sinais (simulação fuzzy)", "ch_loss": "Perda acumulada dos credores, % dos depósitos",
        "inside": "Dentro do controlador neste mês",
        "inside_help": ("1-3: cada entrada vira graus de pertinência (0 a 1) nos seus termos. As regras combinam um "
                        "termo de cada entrada com E (o menor grau). 4: cada regra ativada corta seu triângulo de "
                        "saída na sua força, os cortes são unidos (máximo) e o centro de gravidade da forma "
                        "(centroide) é o novo limite."),
        "rules": "Regras ativadas (força = menor dos três graus)",
        "done": "Simulação concluída.",
        "vol": "Volatilidade (anual)", "util": "Utilização", "rent": "Cobertura de aluguel",
        "try_intro": "Mova as entradas: as quatro etapas do controlador são recalculadas na hora.",
        "result": "LTV máximo",
        "scenarios": {"collapse": "Colapso operacional (RealT 2024-2026): aluguel para, credores saem",
                      "shock": "Choque de mercado (Detroit 2006-2012): preço das casas cai pela metade"},
    },
}

st.set_page_config(page_title="Adaptive LTV", layout="wide")
lang = st.sidebar.radio("Language / Idioma", ["pt", "en"], horizontal=True)
T = TEXT[lang]


@st.cache_resource
def load():
    return default_controller(), {"collapse": operational_collapse(), "shock": market_shock()}


ctrl, SCENARIOS = load()

st.title(T["title"])
st.markdown(T["intro"])
tab_sim, tab_try = st.tabs([T["tab_sim"], T["tab_try"]])


def show_decision(decision, fig_slot, table_slot):
    fig = plot_decision(ctrl, decision)
    fig_slot.pyplot(fig, clear_figure=True)
    plt.close(fig)
    rules = active_rules(decision).rename(columns={"max_ltv_term": "max LTV"})
    rules["strength"] = rules["strength"].round(2)
    table_slot.dataframe(rules, hide_index=True, width="stretch")


# --- manual controller --------------------------------------------------------------------------------------------
with tab_try:
    st.markdown(T["try_intro"])
    a, b, c = st.columns(3)
    v = a.slider(T["vol"], 0.0, 0.30, 0.055, 0.001, format="%.3f")
    u = b.slider(T["util"], 0.70, 1.0, 0.85, 0.005, format="%.3f")
    r = c.slider(T["rent"], 0.0, 1.0, 1.0, 0.01, format="%.2f")
    d = ctrl.decide(v, u, r)
    st.metric(T["result"], f"{d.max_ltv:.1%}")
    show_decision(d, st.empty(), st.empty())
    st.caption(T["inside_help"])


# --- simulation -------------------------------------------------------------------------------------------------
with tab_sim:
    st.sidebar.header(T["scenario"])
    key = st.sidebar.radio(T["scenario"], list(T["scenarios"]), format_func=T["scenarios"].get,
                           label_visibility="collapsed")
    speed = st.sidebar.slider(T["speed"], 0.0, 1.0, 0.25, 0.05)
    st.sidebar.header(T["assumptions"])
    delev = st.sidebar.slider(T["delev"], 0.0, 0.20, env.ASSUMPTIONS["deleverage_rate"][0], 0.01, format="%.2f")
    discount = st.sidebar.slider(T["discount"], 0.40, 0.80, env.ASSUMPTIONS["token_discount"][0], 0.05,
                                 format="%.2f")
    go = st.sidebar.button(T["start"], type="primary", width="stretch")

    m = st.columns(4)
    metric_slots = [c.empty() for c in m]
    c1, c2 = st.columns(2)
    ltv_slot, util_slot = c1.empty(), c2.empty()
    c3, c4 = st.columns(2)
    sig_slot, loss_slot = c3.empty(), c4.empty()
    st.subheader(T["inside"])
    st.caption(T["inside_help"])
    fig_slot = st.empty()
    table_slot = st.empty()

    if go:
        scenario = SCENARIOS[key]
        last = {}

        def fuzzy_policy(v, u, r):
            last["decision"] = ctrl.decide(v, u, r)
            return last["decision"].max_ltv

        with env.assumption("deleverage_rate", delev), env.assumption("token_discount", discount):
            supply0 = env.initial_state(scenario, env.START_LTV).supply
            fixed_run = env.simulate(scenario, env.fixed_policy())
            fuzzy_run = env.simulate(scenario, fuzzy_policy)
            hist = []
            realized = {"fixed": 0.0, "fuzzy": 0.0}
            for (_, rf), (_, rz) in zip(fixed_run, fuzzy_run):
                realized["fixed"] += rf["bad_debt"]
                realized["fuzzy"] += rz["bad_debt"]
                loss_f = 100 * (realized["fixed"] + rf["latent_bad_debt"]) / supply0
                loss_z = 100 * (realized["fuzzy"] + rz["latent_bad_debt"]) / supply0
                hist.append({"date": rz["date"],
                             "ltv fixed": 100 * rf["max_ltv"], "ltv fuzzy": 100 * rz["max_ltv"],
                             "util fixed": 100 * rf["utilization"], "util fuzzy": 100 * rz["utilization"],
                             "volatility": 100 * rz["volatility"], "rent": 100 * rz["rent_coverage"],
                             "loss fixed": loss_f, "loss fuzzy": loss_z})
                df = pd.DataFrame(hist).set_index("date")

                metric_slots[0].metric(T["month"], rz["date"].strftime("%Y-%m"))
                metric_slots[1].metric(T["fuzzy_ltv"], f"{rz['max_ltv']:.1%}")
                metric_slots[2].metric(T["loss_fixed"], f"{loss_f:.1f}%", help=T["loss_help"])
                metric_slots[3].metric(T["loss_fuzzy"], f"{loss_z:.1f}%", help=T["loss_help"])
                with ltv_slot.container():
                    st.markdown(f"**{T['ch_ltv']} (%)**")
                    st.line_chart(df[["ltv fixed", "ltv fuzzy"]], color=["#9a9a9a", "#2a6fdb"], height=220)
                with util_slot.container():
                    st.markdown(f"**{T['ch_util']} (%)**")
                    st.line_chart(df[["util fixed", "util fuzzy"]], color=["#9a9a9a", "#2a6fdb"], height=220)
                with sig_slot.container():
                    st.markdown(f"**{T['ch_signals']} (%)**")
                    st.line_chart(df[["volatility", "rent"]], color=["#eb6834", "#1baf7a"], height=220)
                with loss_slot.container():
                    st.markdown(f"**{T['ch_loss']}**")
                    st.line_chart(df[["loss fixed", "loss fuzzy"]], color=["#9a9a9a", "#2a6fdb"], height=220)
                show_decision(last["decision"], fig_slot, table_slot)
                time.sleep(speed)
        st.success(T["done"])
