import math
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit

# ------- CARICAMENTO DATI -------
file_path = 'N_09_extent_v4.0.csv'
df = pd.read_csv(file_path, skipinitialspace=True)
df.columns = df.columns.str.strip()
df = df[df['area'] > 0] # filtro di sicurezza

years = df['year'].values
areas = df['area'].values
t = years - 1979 # rescaling in modo tale che t=0 nel 1979

# ------- FIT -------

def linear_model(t, A, B):
    return A + B * t

def quadratic_model(t, A, B, C):
    return A + B * t + C * t**2

# ------- lineare -------

popt_lin, pcov_lin = curve_fit(linear_model, t, areas)
A_lin, B_lin = popt_lin
y_pred_lin = linear_model(t, *popt_lin)

# ------- quadratico -------

popt_quad, pcov_quad = curve_fit(quadratic_model, t, areas)
A_quad, B_quad, C_quad = popt_quad
y_pred_quad = quadratic_model(t, *popt_quad)


# ------- STAMPA DEI PLOT -------

def setup_plot(ax, title, ylabel):
    ax.grid(True, which='major', linestyle=':', linewidth=0.8, color='gray', alpha=0.6)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray', alpha=0.4)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15, color='black')
    ax.set_xlabel('Anno', fontsize=12, fontweight='bold', color='black')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', color='black')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


fig0, ax0 = plt.subplots(figsize=(12, 7))
setup_plot(ax0, 'Dati Osservati', 'Sea Ice Area [Milioni di km²]')
ax0.scatter(years, areas, color='teal', alpha=0.6, s=60, edgecolors='white', linewidth=0.8, label='Dati NSIDC (Settembre)')
ax0.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
fig0.tight_layout()

fig1, ax1 = plt.subplots(figsize=(12, 7))
setup_plot(ax1, 'Fit Trend Lineare', 'Sea Ice Area [Milioni di km²]')
ax1.scatter(years, areas, color='teal', alpha=0.3, s=50, edgecolors='none', label='Dati NSIDC (Settembre)')
ax1.plot(years, y_pred_lin, color='tab:red', linewidth=3, label=f'Fit Lineare')


stats_text_lin = (
    f"$y = {A_lin:.2f}  {B_lin:.4f}t$\n"
)
bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="tab:red", alpha=0.9, lw=1.5)
ax1.text(0.02, 0.15, stats_text_lin, transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', bbox=bbox_props, color='black')
ax1.legend(frameon=True, fancybox=True, shadow=True)
fig1.tight_layout()

fig2, ax2 = plt.subplots(figsize=(12, 7))
setup_plot(ax2, 'Fit Trend Quadratico', 'Sea Ice Area [Milioni di km²]')
ax2.scatter(years, areas, color='teal', alpha=0.3, s=50, edgecolors='none', label='Dati NSIDC (Settembre)')
ax2.plot(years, y_pred_quad, color='tab:blue', linewidth=3, label=f'Fit Quadratico')

stats_text_quad = (
    f"$y = {A_quad:.2f}  {B_quad:.4f}t  {C_quad:.5f}t^2$\n"
)
bbox_props_quad = dict(boxstyle="round,pad=0.5", fc="white", ec="tab:blue", alpha=0.9, lw=1.5)
ax2.text(0.02, 0.20, stats_text_quad, transform=ax2.transAxes, fontsize=12,
         verticalalignment='top', bbox=bbox_props_quad, color='black')

ax2.legend(frameon=True, fancybox=True, shadow=True)
fig2.tight_layout()

plt.show()

# ------- STAMPA DEI PARAMETRI DEI FIT -------

print("Parametri lineari (a, b):", popt_lin)
print("Parametri quadratici (a, b, c):", popt_quad)

# ------- statistiche globali sulla sigma -------

# deviazione standard sui dati grezzi (fluttuazione totale dal valore medio)
sigma_raw = np.std(areas, ddof=1)

# deviazione standard dei residui (fluttuazioni dopo il detrending)
residui_quad = areas - y_pred_quad
sigma_resid = np.std(residui_quad, ddof=1)

print("STATISTICHE GLOBALI")
print(f"Std Dev (dati grezzi):   {sigma_raw:.3f} Milioni di km^2")
print(f"Std Dev (residui):  {sigma_resid:.3f} Milioni di km^2")


# ------- analisi mobile -------

window_size = 10
residui_series = pd.Series(residui_quad, index=years)
rolling_sigma = residui_series.rolling(window=window_size, center=True).std()

valid_idx = ~np.isnan(rolling_sigma)
years_valid = years[valid_idx]
sigma_valid = rolling_sigma[valid_idx]
slope_sigma, intercept_sigma = np.polyfit(years_valid, sigma_valid, 1)
trend_sigma = slope_sigma * years_valid + intercept_sigma

print(f"Pendenza trend instabilità: {slope_sigma:.5f}")

# ------- stampa plot -------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False) # Aumentato leggermente l'altezza

setup_plot(ax1, 'Residui dal Trend Quadratico', 'Residui [Milioni di $km^2$]')
ax1.set_xlabel('')
ax1.bar(years, residui_quad, color='gray', alpha=0.6, label='Residui Detrendizzati', zorder=3)
ax1.axhline(0, color='black', linewidth=1, zorder=4)
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

setup_plot(ax2, 'Evoluzione della Variabilità tramite Rolling Window Analysis', 'Deviazione Standard [Milioni di $km^2$]')
ax2.plot(years, rolling_sigma, color='purple', linewidth=2,
         marker='o', markersize=5, label=f'Rolling Std Dev (finestra {window_size} anni)', zorder=3)
ax2.plot(years_valid, trend_sigma, color='darkorange', linestyle='--', linewidth=2,
         label=f'Trend variabilità (pendenza={slope_sigma:.5f})', zorder=3)
ax2.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)

for ax in [ax1, ax2]:
    ax.set_xlim(years.min()-1, years.max()+1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

plt.tight_layout()
plt.show()