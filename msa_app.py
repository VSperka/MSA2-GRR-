import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --- Nastavení vzhledu stránky ---
st.set_page_config(page_title="MSA Analýza", layout="wide")

# --- 1. Postranní panel (Sidebar) pro zadání parametrů ---
st.sidebar.header("Základní parametry")
num_operators = st.sidebar.number_input("Počet operátorů", min_value=1, max_value=10, value=3)
num_parts = st.sidebar.number_input("Počet kusů", min_value=2, max_value=30, value=10)
num_trials = st.sidebar.number_input("Počet opakování", min_value=2, max_value=10, value=2)

# --- Dynamický název aplikace podle počtu operátorů ---
if num_operators == 1:
    study_title = "Studie Typu 3 (Automatizovaný systém)"
    main_metric_label = "Celková chyba (EV)"
else:
    study_title = "Studie Typu 2 (Gage R&R (Metoda ANOVA))"
    main_metric_label = "Gage R&R"

st.title(f"Metodika MSA: {study_title}")
st.write("Zadej parametry studie v levém panelu, uprav naměřená data v tabulce a spusť vyhodnocení.")

# Definice dvou záložek
tab1, tab2 = st.tabs(["Opakovatelnost (Typ 3 / GR&R)", "Analýza stability (Drift)"])

with tab1:

    st.sidebar.header("Specifikace (Toleranční pole)")
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
        
        # 1. KROK: Příprava dat (tento krok ti pravděpodobně chyběl)
        # Přejmenujeme sloupce z české tabulky zpět do angličtiny pro statistický model
        df_model = edited_df.rename(columns={
            "Operátor": "Operator", 
            "Kus": "Part", 
            "Opakování": "Trial", 
            "Naměřená hodnota": "Measurement"
        })
        
        # Očistíme data pro případ, že by po vložení z Excelu vznikly prázdné řádky
        df_model = df_model.dropna(subset=['Measurement'])
        
        # Ujistíme se, že hodnoty jsou čísla (při kopírování z Excelu to občas zlobí)
        df_model['Measurement'] = pd.to_numeric(df_model['Measurement'], errors='coerce')
        
        # 2. KROK: Rozhodnutí podle počtu operátorů
        if num_operators == 1:
            st.info("💡 Byl zvolen 1 operátor: Analýza probíhá jako Studie Typu 3 (automatizovaný systém). Vliv operátora (AV) se neuvažuje a je roven nule.")
            
            # Jednofaktorová ANOVA (pouze vliv kusu)
            model = ols('Measurement ~ C(Part)', data=df_model).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            MS_part = anova_table.loc['C(Part)', 'sum_sq'] / anova_table.loc['C(Part)', 'df']
            MS_rep = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
            
            var_EV = MS_rep
            var_AV = 0.0 # Operátor nehraje roli
            var_GRR = var_EV + var_AV
            var_PV = max(0, (MS_part - MS_rep) / num_trials)
            var_TV = var_GRR + var_PV
    
        else:
            # Původní dvoufaktorová ANOVA s interakcí pro více operátorů
            model = ols('Measurement ~ C(Part) * C(Operator)', data=df_model).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            MS_part = anova_table.loc['C(Part)', 'sum_sq'] / anova_table.loc['C(Part)', 'df']
            MS_op = anova_table.loc['C(Operator)', 'sum_sq'] / anova_table.loc['C(Operator)', 'df']
            MS_int = anova_table.loc['C(Part):C(Operator)', 'sum_sq'] / anova_table.loc['C(Part):C(Operator)', 'df']
            MS_rep = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
            
            var_EV = MS_rep
            var_op = max(0, (MS_op - MS_int) / (num_parts * num_trials))
            var_int = max(0, (MS_int - MS_rep) / num_trials)
            var_AV = var_op + var_int
            
            var_GRR = var_EV + var_AV
            var_PV = max(0, (MS_part - MS_int) / (num_operators * num_trials))
            var_TV = var_GRR + var_PV
        
        # 3. KROK: Výpočet procent
        pct_EV = (np.sqrt(var_EV) / np.sqrt(var_TV)) * 100 if var_TV > 0 else 0
        pct_AV = (np.sqrt(var_AV) / np.sqrt(var_TV)) * 100 if var_TV > 0 else 0
        pct_GRR = (np.sqrt(var_GRR) / np.sqrt(var_TV)) * 100 if var_TV > 0 else 0
        pct_PV = (np.sqrt(var_PV) / np.sqrt(var_TV)) * 100 if var_TV > 0 else 0
        
        ndc = max(1, int(np.sqrt(2) * (np.sqrt(var_PV) / np.sqrt(var_GRR)))) if var_GRR > 0 else 999
        
        # Výpočet % z tolerance
        tolerance = usl - lsl
        if tolerance > 0:
            pct_tol_GRR = ((np.sqrt(var_GRR) * 6) / tolerance) * 100
        else:
            pct_tol_GRR = 0.0
    
        # --- 4. Zobrazení výsledků formou metrik ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{main_metric_label} (% ze studie)", f"{pct_GRR:.2f} %", 
                    delta="Vyhovuje" if pct_GRR <= 10 else "Podmíněně" if pct_GRR <= 30 else "Nevyhovuje",
                    delta_color="inverse")
        
        col2.metric("Opakovatelnost (EV)", f"{pct_EV:.2f} %")
        
        # Pokud je 1 operátor, AV je nula, tak ať to uživatele nemate, můžeme to i skrýt, nebo nechat 0%
        col3.metric("Reprodukovat. (AV)", f"{pct_AV:.2f} %")
        
        col4.metric(f"{main_metric_label} (% z tolerance)", f"{pct_tol_GRR:.2f} %",
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
        if num_operators > 1:
            sns.pointplot(data=df_model, x='Part', y='Measurement', hue='Operator', ax=axes[1], markers='o', errorbar=None)
            axes[1].set_title("Průměry naměřených hodnot (Interakce)")
            axes[1].set_xlabel("Kusy")
            axes[1].set_ylabel("Naměřená hodnota")
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, linestyle=':', alpha=0.7)
        else:
            # Pokud je 1 operátor, vykreslíme místo toho boxplot rozptylu jednotlivých kusů
            sns.boxplot(data=df_model, x='Part', y='Measurement', ax=axes[1], palette='Set2')
            axes[1].set_title("Variabilita naměřených hodnot pro jednotlivé kusy")
            axes[1].set_xlabel("Kusy")
            axes[1].set_ylabel("Naměřená hodnota")
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig) # Streamlit funkce pro zobrazení matplotlib grafu

with tab2:
    st.header("Analýza stability měřicího systému (Drift)")
    st.write("Zadej periodicky naměřené hodnoty jednoho referenčního kusu (Master Part).")

    # Nové zadávací pole pro počet měření
    num_drift_points = st.number_input("Počet měření etalonu", min_value=2, max_value=200, value=20, step=1)

    # 1. Příprava datové tabulky pro drift
    @st.cache_data
    def generate_drift_data(n_points):
        np.random.seed(10)
        # Simulace stabilního procesu s mírným šumem
        base_values = np.random.normal(100, 0.5, n_points)
        data = [{"Čas měření": f"Měření {i+1}", "Hodnota etalonu": round(val, 2)} for i, val in enumerate(base_values)]
        return pd.DataFrame(data)

    # Volání funkce s dynamickou proměnnou místo pevného čísla
    df_drift_base = generate_drift_data(num_drift_points)
    edited_drift_df = st.data_editor(df_drift_base, use_container_width=True, hide_index=True)

    # 2. Výpočet regulačních mezí (I-MR)
    if st.button("Vyhodnotit stabilitu", type="primary", key="btn_drift"):
        
        # Převedení na čísla a odstranění prázdných řádků
        drift_data = pd.to_numeric(edited_drift_df['Hodnota etalonu'], errors='coerce').dropna().values
        
        if len(drift_data) < 2:
            st.warning("Pro analýzu driftu jsou potřeba alespoň 2 naměřené hodnoty.")
        else:
            # Výpočet průměru (osa X)
            mean_x = np.mean(drift_data)
            
            # Výpočet klouzavého rozpětí (Moving Range - rozdíly mezi sousedními body)
            mr = np.abs(np.diff(drift_data))
            mean_mr = np.mean(mr)
            
            # Výpočet UCL a LCL pro jednotlivé hodnoty (konstanta 2.66 pro n=2)
            ucl_x = mean_x + 2.66 * mean_mr
            lcl_x = mean_x - 2.66 * mean_mr

            # 3. Zobrazení metrik
            col1, col2, col3 = st.columns(3)
            col1.metric("Průměrná hodnota (X-bar)", f"{mean_x:.2f}")
            col2.metric("Horní mez (UCL)", f"{ucl_x:.2f}")
            col3.metric("Spodní mez (LCL)", f"{lcl_x:.2f}")

            # 4. Vykreslení regulačního diagramu
            fig_drift, ax_drift = plt.subplots(figsize=(12, 5))
            
            # Vynesení naměřených bodů
            ax_drift.plot(drift_data, marker='o', linestyle='-', color='b', label='Naměřená hodnota')
            
            # Vynesení regulačních mezí a střední osy
            ax_drift.axhline(mean_x, color='green', linestyle='-', label='Průměr')
            ax_drift.axhline(ucl_x, color='red', linestyle='--', label='UCL / LCL')
            ax_drift.axhline(lcl_x, color='red', linestyle='--')
            
            # Zvýraznění bodů mimo regulační meze (Out of Control)
            out_of_control_idx = np.where((drift_data > ucl_x) | (drift_data < lcl_x))[0]
            if len(out_of_control_idx) > 0:
                ax_drift.plot(out_of_control_idx, drift_data[out_of_control_idx], 'ro', markersize=10, label='Mimo meze')
                st.error(f"Detekována nestabilita! Počet bodů mimo regulační meze: {len(out_of_control_idx)}. Měřidlo vykazuje signifikantní drift nebo zvláštní příčinu variability.")
            else:
                st.success("Měřicí systém je statisticky stabilní v čase. Žádné body neleží mimo regulační meze.")

            ax_drift.set_title("Regulační diagram jednotlivých hodnot (Drift)")
            ax_drift.set_xlabel("Pořadí měření")
            ax_drift.set_ylabel("Naměřená hodnota")
            ax_drift.legend()
            ax_drift.grid(True, linestyle=':', alpha=0.7)
            
            st.pyplot(fig_drift)
