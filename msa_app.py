import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --- Nastavení vzhledu stránky ---
st.set_page_config(page_title="Gage R&R Analýza", layout="wide")
st.title("Metodika MSA: Gage R&R (Metoda ANOVA)")
st.write("Zadej parametry studie v levém panelu, uprav naměřená data v tabulce a spusť vyhodnocení.")

# --- 1. Postranní panel (Sidebar) pro zadání parametrů ---
st.sidebar.header("Základní parametry")
num_operators = st.sidebar.number_input("Počet operátorů", min_value=2, max_value=10, value=3)
num_parts = st.sidebar.number_input("Počet kusů", min_value=2, max_value=30, value=10)
num_trials = st.sidebar.number_input("Počet opakování", min_value=2, max_value=10, value=2)

st.sidebar.header("Specifikace (Toleranční pole)")
nominal_value = st.sidebar.number_input("Jmenovitá (cílová) hodnota", value=100.0)
lsl = st.sidebar.number_input("Spodní mez (LSL)", value=90.0)
usl = st.sidebar.number_input("Horní mez (USL)", value=110.0)

# --- 2. Příprava interaktivní tabulky ---
# Vytvoříme funkci, která vygeneruje strukturu tabulky s ukázkovými daty
@st.cache_data # Streamlit si data zapamatuje, dokud nezměníš parametry
def generate_base_data(ops, parts, trials):
    np.random.seed(42)
    data = []
    # Generování realistických fiktivních dat
    part_means = np.random.normal(100, 5, parts)
    op_biases = np.random.normal(0, 0.5, ops)
    
    for p in range(parts):
        for o in range(ops):
            for t in range(trials):
                # Kus + vliv operátora + náhodná chyba měřidla
                val = part_means[p] + op_biases[o] + np.random.normal(0, 0.2)
                data.append({
                    "Operátor": f"Operátor {o+1}",
                    "Kus": f"Kus {p+1}",
                    "Opakování": f"Pokus {t+1}",
                    "Naměřená hodnota": round(val, 2)
                })
    return pd.DataFrame(data)

# Zobrazení tabulky, kterou může uživatel přímo na stránce upravovat
st.subheader("Tabulka naměřených dat")
df_base = generate_base_data(num_operators, num_parts, num_trials)
edited_df = st.data_editor(df_base, use_container_width=True, hide_index=True)

# --- 3. Výpočet a vyhodnocení ---
# Tlačítko, které spustí celou matematiku
if st.button("Vyhodnotit Gage R&R", type="primary"):
    
    st.markdown("---")
    st.header("Výsledky analýzy")
    
    # Přejmenujeme sloupce zpět do angličtiny, aby to vzal statistický model
    df_model = edited_df.rename(columns={
        "Operátor": "Operator", 
        "Kus": "Part", 
        "Opakování": "Trial", 
        "Naměřená hodnota": "Measurement"
    })
    
    # Sestavení a výpočet ANOVA modelu
    model = ols('Measurement ~ C(Part) * C(Operator)', data=df_model).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Extrakce středních čtverců (Mean Squares)
    MS_part = anova_table.loc['C(Part)', 'sum_sq'] / anova_table.loc['C(Part)', 'df']
    MS_op = anova_table.loc['C(Operator)', 'sum_sq'] / anova_table.loc['C(Operator)', 'df']
    MS_int = anova_table.loc['C(Part):C(Operator)', 'sum_sq'] / anova_table.loc['C(Part):C(Operator)', 'df']
    MS_rep = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
    
    # Výpočet rozptylů (Variance)
    var_EV = MS_rep # Opakovatelnost (vybavení)
    
    # Reprodukovatelnost (operátoři + interakce)
    var_op = max(0, (MS_op - MS_int) / (num_parts * num_trials))
    var_int = max(0, (MS_int - MS_rep) / num_trials)
    var_AV = var_op + var_int
    
    var_GRR = var_EV + var_AV # Celá chyba měření
    var_PV = max(0, (MS_part - MS_int) / (num_operators * num_trials)) # Variabilita kusů
    var_TV = var_GRR + var_PV # Celková variabilita procesu
    
    # Výpočet % z celkové variace
    pct_EV = (np.sqrt(var_EV) / np.sqrt(var_TV)) * 100
    pct_AV = (np.sqrt(var_AV) / np.sqrt(var_TV)) * 100
    pct_GRR = (np.sqrt(var_GRR) / np.sqrt(var_TV)) * 100
    pct_PV = (np.sqrt(var_PV) / np.sqrt(var_TV)) * 100
    
    # Počet rozlišitelných kategorií (NDC)
    ndc = max(1, int(np.sqrt(2) * (np.sqrt(var_PV) / np.sqrt(var_GRR)))) if var_GRR > 0 else 999
    
    # --- 4. Zobrazení výsledků formou metrik a tabulky ---
    # Výpočet % z tolerance (Precision to Tolerance)
    tolerance = usl - lsl
    if tolerance > 0:
        pct_tol_GRR = ((np.sqrt(var_GRR) * 6) / tolerance) * 100
    else:
        pct_tol_GRR = 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gage R&R (% ze studie)", f"{pct_GRR:.2f} %", 
                delta="Vyhovuje" if pct_GRR <= 10 else "Podmíněně" if pct_GRR <= 30 else "Nevyhovuje",
                delta_color="inverse")
    
    col2.metric("Opakovatelnost (EV)", f"{pct_EV:.2f} %")
    
    col3.metric("Reprodukovat. (AV)", f"{pct_AV:.2f} %")
    
    col4.metric("Gage R&R (% z tolerance)", f"{pct_tol_GRR:.2f} %",
                delta="Vyhovuje" if pct_tol_GRR <= 10 else "Podmíněně" if pct_tol_GRR <= 30 else "Nevyhovuje",
                delta_color="inverse")
    
    st.write(f"**Počet rozlišitelných kategorií (NDC):** {ndc} (Požadavek: ≥ 5)")
    
    # --- 5. Vykreslení grafů ---
    st.subheader("Grafické vyhodnocení")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graf 1: Složky rozptylu
    komponenty = ['EV (Opakovatelnost)', 'AV (Reprodukovatelnost)', 'Gage R&R', 'PV (Mezi kusy)']
    hodnoty_pct = [pct_EV, pct_AV, pct_GRR, pct_PV]
    
    sns.barplot(x=komponenty, y=hodnoty_pct, ax=axes[0], palette='Blues_r')
    axes[0].set_title("Příspěvek komponent k celkové variaci")
    axes[0].set_ylabel("% Variace")
    axes[0].axhline(y=10, color='green', linestyle='--', label='10% (Ideální)')
    axes[0].axhline(y=30, color='red', linestyle='--', label='30% (Limit)')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=15)
    
    # Graf 2: Interakce (Průměry kusů vs. Operátoři)
    sns.pointplot(data=df_model, x='Part', y='Measurement', hue='Operator', ax=axes[1], markers='o', errorbar=None)
    axes[1].set_title("Průměry naměřených hodnot (Interakce)")
    axes[1].set_xlabel("Kusy")
    axes[1].set_ylabel("Naměřená hodnota")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(fig) # Streamlit funkce pro zobrazení matplotlib grafu
