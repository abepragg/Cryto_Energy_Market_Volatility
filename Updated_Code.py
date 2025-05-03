import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy.stats import rankdata
from pyvinecopulib import Bicop, BicopFamily, FitControlsBicop
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import itertools

st.set_page_config(layout="wide")
st.title("Crypto & Energy Market Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    # log_returns = pd.read_excel(uploaded_file, sheet_name="Sheet4_LogReturns", parse_dates=['date'])
    log_returns = pd.read_excel(uploaded_file, parse_dates=['date'])
    log_returns = log_returns.drop(columns=['index'], errors='ignore').dropna()
    log_returns.set_index('date', inplace=True)

    cryptos = ['BTC', 'ETH', 'ADA', 'LTC', 'ETC', 'XMR', 'DOT', 'SOL']
    gold = 'XAU'
    all_assets = cryptos + [gold]
    energy_options = [col for col in log_returns.columns if col not in all_assets]
    energy_selected = st.sidebar.selectbox("Select an energy market", energy_options)

    def fit_garch(series):
        model = arch_model(series, vol='Garch', p=1, q=1)
        res = model.fit(disp='off')
        return res.conditional_volatility

    # Section 1: Distributions
    st.header(f"1. Return Distributions vs {energy_selected}")
    for asset in all_assets:
        df = log_returns[[asset, energy_selected]].dropna()
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        axs[0, 0].plot(df.index, df[asset])
        axs[0, 0].set_title(f'{asset} Returns')
        sns.histplot(df[asset], kde=True, ax=axs[0, 1])
        axs[0, 1].set_title(f'{asset} Return Distribution')
        axs[1, 0].plot(df.index, df[energy_selected])
        axs[1, 0].set_title(f'{energy_selected} Returns')
        sns.histplot(df[energy_selected], kde=True, ax=axs[1, 1])
        axs[1, 1].set_title(f'{energy_selected} Return Distribution')
        st.pyplot(fig)

    # Section 2: Low Volatility Performance
    st.header(f"2. Low Volatility Performance")
    energy_vol = fit_garch(log_returns[energy_selected])
    log_returns['EnergyVol'] = energy_vol
    low_vol_dates = energy_vol[energy_vol <= np.percentile(energy_vol.dropna(), 5)].index

    results = []
    for asset in all_assets:
        asset_returns = log_returns.loc[low_vol_dates, asset].dropna()
        if len(asset_returns) > 5:
            avg_return = asset_returns.mean()
            volatility = asset_returns.std()
            sharpe = avg_return / volatility * np.sqrt(252)
            results.append({
                'Asset': asset,
                'Average Return': round(avg_return, 4),
                'Volatility': round(volatility, 4),
                'Sharpe Ratio': round(sharpe, 2)
            })

    summary_df = pd.DataFrame(results).set_index('Asset')
    st.dataframe(summary_df)

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    for asset in all_assets:
        series = log_returns.loc[low_vol_dates, asset].dropna()
        if not series.empty:
            ax3.plot(series.index, series.values, label=asset)
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_title(f"Returns During Low {energy_selected} Volatility")
    ax3.legend()
    st.pyplot(fig3)

    # Section 3: Crash Detection
    st.header("3. Crash Detection")
    cumulative_energy = log_returns[energy_selected].cumsum()
    rolling_max = cumulative_energy.cummax()
    drawdown = cumulative_energy - rolling_max
    drawdown_pct = drawdown / rolling_max


    threshold = -0.20  # 20% drop is widely used as a market crash/stress threshold



    in_crash = False
    windows = []
    start = None

    for date, dd in drawdown_pct.items():
        if dd < threshold and not in_crash:
            start = date
            in_crash = True
        elif dd >= 0 and in_crash:
            end = date
            if (end - start).days >= 10:  # Filter out short crashes
                windows.append((start, end))
            in_crash = False


    crash_results = []
    for start, end in windows:
        window_data = log_returns.loc[start:end]
        if window_data.empty:
            continue
        result = {
            'Energy Crash Start': start,
            'Recovery Date': end,
            'Duration': f"{(end - start).days} days",
            f'{energy_selected} Drop': f"{(cumulative_energy.loc[end] - cumulative_energy.loc[start]) / abs(cumulative_energy.loc[start]) * 100:.2f}%"
        }
        for asset in all_assets:
            change = window_data[asset].sum()
            result[f'{asset} Change'] = f"{change * 100:.2f}%" + (" ↑" if change > 0 else " ↓")
        crash_results.append(result)

        fig4, ax4 = plt.subplots(figsize=(12, 5))
        for asset in all_assets:
            ax4.plot(window_data.index, window_data[asset].cumsum(), label=asset)
        ax4.plot(window_data.index, window_data[energy_selected].cumsum(), label=energy_selected, color='black', linewidth=2, linestyle='--')
        ax4.plot(window_data.index, window_data['XAU'].cumsum(), label='Gold (XAU)', linestyle='dotted')  # Make gold visible
        ax4.axhline(0, color='gray', linestyle='--')
        ax4.set_title(f"Crypto & Gold Performance During {energy_selected} Crash: {start} → {end}")
        ax4.legend()
        st.pyplot(fig4)

    crash_df = pd.DataFrame(crash_results)
    st.dataframe(crash_df)

    # Section 4a: Copula Analysis (General Dependence)
    st.header("4a. Copula Analysis (General Dependence)")
    general_results = []

    for crypto in cryptos:
        df = log_returns[[crypto, energy_selected]].dropna()
        u = rankdata(df[crypto]) / (len(df) + 1)
        v = rankdata(df[energy_selected]) / (len(df) + 1)
        data = np.column_stack([u, v])

        try:
            controls = FitControlsBicop(family_set=[
                BicopFamily.clayton, BicopFamily.tll
            ])
            cop = Bicop()
            cop.select(data, controls=controls)
            selected_family = cop.family.name

            tau = cop.tau
            rho = cop.parameters[0] if cop.family == BicopFamily.tll else "N/A"

            lambda_L = np.mean((u < 0.05) & (v < 0.05))
            lambda_U = np.mean((u > 0.95) & (v > 0.95))

            general_results.append({
                'Crypto': crypto,
                'Selected Copula': selected_family,
                'Kendall Tau (τ)': round(tau, 6),
                'Rho (ρ)': round(rho, 6) if rho != "N/A" else "N/A",
                'Lower Tail (λL)': round(lambda_L, 6),
                'Upper Tail (λU)': round(lambda_U, 6),
            })
        except Exception as e:
            st.warning(f"Copula error for {crypto}: {e}")

    general_df = pd.DataFrame(general_results)
    # Clean up columns before displaying
    if general_df['Rho (ρ)'].replace("N/A", np.nan).dropna().empty:
        general_df.drop(columns=['Rho (ρ)'], inplace=True)

    if general_df['Kendall Tau (τ)'].replace("N/A", np.nan).dropna().eq(0).all():
        general_df.drop(columns=['Kendall Tau (τ)'], inplace=True)
    st.dataframe(general_df)

    st.markdown("""
                    *Classification Criteria*:  
                    - *🟢 Hedge*: τ < -0.1 & λₗ < 0.1 (Baur & Lucey, 2010) 
                    - *🔵 Safe Haven*: λₗ < 0.02 & τ < 0.1 (Baur & McDermott, 2016)  
                    - *🟠 Safe Haven Candidate*: λₗ < 0.1 & τ ≈ 0  
                    - *🟡 Diversifier*: τ < 0 & λₗ < 0.2 (Kroner & Ng, 1998) 
                    
                    """)

    # Section 4b: Crash Explorer
    st.header("4b. Crash Explorer")

    if not crash_df.empty:
        crash_copula_results = []

        for asset in cryptos + ['XAU']:
            # Plot returns with crash window highlights
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(log_returns[asset], label=f"{asset} Returns", color='blue')
            ax.plot(log_returns[energy_selected], label=f"{energy_selected} Returns", color='orange')
            for start, end in windows:
                ax.axvspan(start, end, color='red', alpha=0.2)
            ax.set_title(f"{asset} vs {energy_selected} with Crash Periods Highlighted")
            ax.legend()
            st.pyplot(fig)

            for start, end in windows:
                window_data = log_returns.loc[start:end, [asset, energy_selected]].dropna()
                if len(window_data) < 10:
                    continue

                try:
                    u = rankdata(window_data[asset]) / (len(window_data) + 1)
                    v = rankdata(window_data[energy_selected]) / (len(window_data) + 1)
                    data_uv = np.column_stack([u, v])

                    controls = FitControlsBicop(family_set=[BicopFamily.clayton, BicopFamily.tll])
                    cop = Bicop()
                    cop.select(data_uv, controls=controls)
                    selected_family = cop.family.name

                    tau = cop.tau
                    rho = cop.parameters[0] if cop.family == BicopFamily.tll else "N/A"
                    lambda_L = np.mean((u < 0.05) & (v < 0.05))
                    lambda_U = np.mean((u > 0.95) & (v > 0.95))

                    lambda_L_empirical = np.mean((u < 0.05) & (v < 0.05))
                    # Calculate theoretical λₗ for Clayton copula
                    # Calculate theoretical λₗ for Clayton copula
                    if cop.family == BicopFamily.clayton:
                        try:
                            # Extract theta safely (handles array and scalar cases)
                            theta = float(cop.parameters[0]) if isinstance(cop.parameters, np.ndarray) else float(cop.parameters)
                            lambda_L_theoretical = 2**(-1/theta) if theta > 0 else np.nan
                        except (IndexError, TypeError, ValueError) as e:
                            st.warning(f"Theta extraction failed for {asset}: {str(e)}")
                            lambda_L_theoretical = np.nan
                    else:
                        lambda_L_theoretical = np.nan

                    # Safe rounding function
                    def safe_round(value, decimals=6):
                        if isinstance(value, (np.ndarray, list)):
                            try:
                                return round(float(value[0]), decimals) if len(value) > 0 else "N/A"
                            except (IndexError, TypeError):
                                return "N/A"
                        elif np.isnan(value):
                            return "N/A"
                        try:
                            return round(float(value), decimals)
                        except (TypeError, ValueError):
                            return "N/A"

                    # Get actual returns for asset and energy
                    asset_return = window_data[asset].sum()
                    energy_return = window_data[energy_selected].sum()

                    # # Classification logic (updated with directional behavior)
                    # if tau < 0.1 and lambda_L < 0.05:
                    #     classification = '🟡 Diversifier'
                    # elif tau < -0.1 or lambda_L < 0.02:
                    #     classification = '🟢 Hedge'
                    # elif abs(tau) < 0.1 and lambda_L < 0.02:
                    #     classification = '🔵 Safe Haven'
                    # else:
                    #     classification = '🔴 Not Protective'



                    # Classification logic
                    if tau < -0.1 and lambda_L_empirical < 0.1:
                        classification = '🟢 Hedge'
                    elif lambda_L_empirical < 0.02 and abs(tau) < 0.1:
                        classification = '🔵 Safe Haven'
                    elif abs(tau) < 0.1 and 0.05 <= lambda_L_empirical < 0.1:  # New condition
                        classification = '🟠 Safe Haven Candidate'
                    elif tau < 0 and lambda_L_empirical < 0.2:
                        classification = '🟡 Diversifier'
                    else:
                        classification = '🔴 Not Protective'

                    crash_copula_results.append({
                        'Asset': asset,
                        'Crash Window': f"{start.date()} → {end.date()}",
                        'Selected Copula': selected_family,
                        'Kendall Tau (τ)': round(tau, 6),
                        'Rho (ρ)': round(rho, 6) if rho != "N/A" else "N/A",
                        'Lower Tail (λL)': round(lambda_L, 6),
                        'Upper Tail (λU)': round(lambda_U, 6),
                        'Lower Tail (λL Empirical)': round(lambda_L_empirical, 6),
                        'Lower Tail (λL Theoretical)': safe_round(lambda_L_theoretical),
                        'Classification': classification
                    })

                except Exception as e:
                    st.warning(f"Copula error for {asset} during {start.date()} → {end.date()}: {e}")

        crash_copula_df = pd.DataFrame(crash_copula_results)
        if crash_copula_df['Rho (ρ)'].replace("N/A", np.nan).dropna().empty:
            crash_copula_df.drop(columns=['Rho (ρ)'], inplace=True)

        if crash_copula_df['Kendall Tau (τ)'].replace("N/A", np.nan).dropna().eq(0).all():
            crash_copula_df.drop(columns=['Kendall Tau (τ)'], inplace=True)
        st.dataframe(crash_copula_df)

    else:
        st.warning("No crash periods detected with the current threshold.")

        # ----------------- 📊 5. Crypto Asset Classification bar graph ----------------- #

    st.header("5. Crypto Asset Classification Bar Graph")

    if not crash_copula_df.empty:
        st.subheader(f"Classification Frequency During {energy_selected} Crash Periods")

        classification_summary = crash_copula_df.copy()
        classification_summary["Classification"] = classification_summary["Classification"].str.extract(r'(\w+ \w+|\w+)$')[0]
        asset_grouped = classification_summary.groupby(['Asset', 'Classification']).size().reset_index(name='Count')

        # Define color mapping
        color_mapping = {
            "Hedge": "#2ca02c",               # Green
            "Safe Haven": "#1f77b4",          # Blue
            "Diversifier": "#f1c40f",         # Yellow
            "Safe Haven Candidate": "#ff7f0e",# Orange
            "Not Protective": "#d62728"       # Red
        }


        # Apply custom color palette
        palette = {key: color_mapping[key] for key in asset_grouped["Classification"].unique() if key in color_mapping}

        fig5, ax5 = plt.subplots(figsize=(12, 6))
        sns.barplot(data=asset_grouped, x='Asset', y='Count', hue='Classification', ax=ax5, palette=palette)
        ax5.set_title(f' Asset Classification During {energy_selected} Crash Periods by Crypto')
        ax5.set_xlabel('Asset')
        ax5.set_ylabel('Number of Crash Periods')
        ax5.grid(axis='y', linestyle='--', alpha=0.6)
        ax5.legend(title="Classification")
        st.pyplot(fig5)

        # fig5, ax5 = plt.subplots(figsize=(12, 6))
        # sns.barplot(data=asset_grouped, x='Asset', y='Count', hue='Classification', ax=ax5)
        # ax5.set_title(f'Asset Classification During {energy_selected} Crash Periods by Crypto')
        # ax5.set_xlabel('Asset')
        # ax5.set_ylabel('Number of Crash Periods')
        # ax5.grid(axis='y', linestyle='--', alpha=0.6)
        # st.pyplot(fig5)
    else:
        st.info("No classification data available to visualize.")

    # ----------------- 💰 6. Crash-Aware Portfolio Suggestion ----------------- #

    st.header("6. Crash-Aware Portfolio Suggestion")

    # Investor input
    st.subheader("Investor Preferences")
    total_investment = st.number_input("Total Capital ($)", value=100000.0, min_value=1000.0)
    risk_percent = st.slider("What % of your capital do you want to risk?", 0, 100, 40)
    risk_capital = (risk_percent / 100) * total_investment

    classification_df = pd.DataFrame(crash_copula_results)

    # Clean the Classification column
    classification_df["Classification"] = classification_df["Classification"].str.extract(r'(\w+ \w+|\w+)$')[0]

    # Count classification frequencies per asset
    classification_counts = classification_df.groupby("Asset")["Classification"].value_counts().unstack().fillna(0)

    # Define how to compute majority classification
    def majority_label(row):
        total = row.sum()
        if row.get("Not Protective", 0) > total / 2:
            return "Not Protective"
        elif row.get("Hedge", 0) > 0:
            return "Hedge"
        elif row.get("Safe Haven", 0) > 0:
            return "Safe Haven"
        elif row.get("Safe Haven Candidate", 0) > 0:
            return "Safe Haven Candidate"
        elif row.get("Diversifier", 0) > 0:
            return "Diversifier"
        return "Not Protective"

    classification_counts["Majority Classification"] = classification_counts.apply(majority_label, axis=1)

    # Merge majority classification back into original DataFrame
    classification_df = classification_df.merge(
        classification_counts["Majority Classification"],
        left_on="Asset", right_index=True
    )

    pow_assets = ['BTC', 'LTC', 'ETC', 'XMR']
    pos_assets = ['ADA', 'ETH', 'DOT', 'SOL']
    gold_asset = 'XAU'
    energy_asset = energy_selected

    # Function to select best asset
    def select_best_asset(candidates, class_df, class_priority=["Hedge", "Safe Haven", "Safe Haven Candidate", "Diversifier"]):
        # Group by asset and classification
        grouped = (
            class_df[class_df["Asset"].isin(candidates)]
            .groupby(["Asset", "Classification"])
            .size()
            .unstack(fill_value=0)
        )

        for cls in class_priority:
            if cls in grouped.columns:
                eligible_assets = grouped[grouped[cls] > 0]
                if not eligible_assets.empty:
                    # Select asset with highest count for this classification
                    best = eligible_assets[cls].idxmax()
                    return {
                        "Asset": best,
                        "Majority Classification": cls,
                        "Count": eligible_assets.loc[best, cls]
                    }

        # Fallback: return None or any asset with best overall count
        return None




    # ----------------- 🔷 6a: Crypto Portfolio ----------------- #
    st.subheader("6a. Portfolio: PoW + PoS + Energy Market")

    pow_best = select_best_asset(pow_assets, classification_df)
    pos_best = select_best_asset(pos_assets, classification_df)



    energy_allocation_5a = risk_capital * 0.3
    crypto_allocation_total = risk_capital - energy_allocation_5a
    selected_cryptos = [asset for asset in [pow_best, pos_best] if asset is not None]

    portfolio_5a = []

    if not selected_cryptos:
        st.warning("No suitable crypto assets found. Allocate entirely to Energy Market or reconsider your preferences.")
    else:
        allocation_per_crypto = crypto_allocation_total / len(selected_cryptos)
        for crypto in selected_cryptos:
            portfolio_5a.append({
                "Asset": crypto["Asset"],
                "Type": "Crypto (PoW)" if crypto["Asset"] in pow_assets else "Crypto (PoS)",
                "Classification": crypto["Majority Classification"],
                "Allocation ($)": round(allocation_per_crypto, 2)
            })

    portfolio_5a.append({
        "Asset": energy_asset,
        "Type": "Energy Market",
        "Classification": "N/A",
        "Allocation ($)": round(energy_allocation_5a, 2)
    })

    st.dataframe(pd.DataFrame(portfolio_5a))

    # ----------------- 🟡 6b: Gold Portfolio ----------------- #
    st.subheader("6b. Portfolio: Gold + Energy Market")

    gold_best = select_best_asset([gold_asset], classification_df)

    gold_allocation = risk_capital * 0.6 if gold_best is not None else 0
    energy_allocation_5b = risk_capital - gold_allocation

    portfolio_5b = []

    if gold_best is not None:
        portfolio_5b.append({
            "Asset": gold_best["Asset"],
            "Type": "Gold",
            "Classification": gold_best["Majority Classification"],
            "Allocation ($)": round(gold_allocation, 2)
        })
    else:
        st.warning("Gold was not classified as suitable. Allocation entirely to Energy Market.")

    portfolio_5b.append({
        "Asset": energy_asset,
        "Type": "Energy Market",
        "Classification": "N/A",
        "Allocation ($)": round(energy_allocation_5b, 2)
    })

    st.dataframe(pd.DataFrame(portfolio_5b))

    # ----------------- 💡 6c: Insights ----------------- #
    st.subheader("6c. Insights & Recommendations")

    def summarize_insights(p5a, p5b):
        insights = []

        for p, name in zip([p5a, p5b], ['Crypto Portfolio', 'Gold Portfolio']):
            protective_assets = [i for i in p if i['Classification'] in ["Hedge", "Safe Haven"]]
            diversifiers = [i for i in p if i['Classification'] == "Diversifier"]

            if protective_assets:
                insights.append(f" {name} contains protective assets ({', '.join(a['Asset'] for a in protective_assets)}).")
            if diversifiers:
                insights.append(f" {name} offers diversification via {', '.join(a['Asset'] for a in diversifiers)}.")

        if not insights:
            insights.append("⚠️ Neither portfolio contains strongly protective or diversifying assets based on current classification.")

        insights.append("\n💬 Select the portfolio that aligns best with your risk profile and investment goals.")
        insights.append("\n ⚠️ Subject to Markest Risk")

        return "\n".join(insights)

    st.markdown(summarize_insights(portfolio_5a, portfolio_5b))
