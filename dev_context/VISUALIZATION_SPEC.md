# VISUALIZATION SPEC: Aesthetic & Library Standards

**Objective**: Ensure all output is "Boardroom Ready" and consistent.
**Design Philosophy**: "Clean, Professional, Data-First". No rainbow charts.

## 1. Libraries (Strict Enforcement)

*   **Static Reports (PDF/PNG)**: Use **Matplotlib** (with strict styling).
*   **Interactive Dashboards**: Use **Plotly**.
*   **Seaborn**: Allowed only for statistical distributions (KDE, Boxplots) on top of Matplotlib.

---

## 2. Institutional Color Palette (The "Jane Street" Theme)

Do not rely on default matplotlib colors (`tab:blue`, `tab:orange`). Use these specific Hex codes.

### A. Semantic Colors
| Meaning | Hex Code | Visual | Usage |
|:--------|:---------|:-------|:------|
| **Profit / Calm / Up** | `#2ECC71` | Green | Normal regimes, Positive PnL. |
| **Loss / Stress / Down** | `#E74C3C` | Red | Volatile regimes, Drawdowns. |
| **Neutral / Text** | `#2C3E50` | Navy | Axis labels, Price lines. |
| **Warning / Breach** | `#E67E22` | Orange | VaR Lines, Near-Miss events. |

### B. Interface Colors
| Element | Hex Code | Note |
|:--------|:---------|:-----|
| **Background** | `#FFFFFF` | Always White. No Dark Mode for reports. |
| **Grid Lines** | `#ECF0F1` | Very subtle gray. |
| **Spines/Borders** | `#BDC3C7` | Medium gray. |
| **Highlight** | `#8E44AD` | Purple (for CVaR or secondary emphasis). |

### C. Implementation Example
```python
COLORS = {
    'price':   '#2C3E50',
    'calm':    '#2ECC71',
    'stress':  '#E74C3C',
    'grid':    '#ECF0F1',
    'text':    '#2C3E50'
}
```

---

## 3. Layout Standards

### A. Mandatory Elements
Every plot MUST have:
1.  **Grid**: `alpha=0.5`, `color='#ECF0F1'`, `linestyle='-'`.
2.  **Spines**: Remove Top and Right spines.
    ```python
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ```
3.  **Title**: Bold, Left-aligned.
    ```python
    ax.set_title("Figure 1. Portfolio Performance", fontweight='bold', loc='left')
    ```
4.  **Watermark / Timestamp**: Bottom right corner, small font.
    ```python
    fig.text(0.95, 0.02, f"Generated: {datetime.now()}", ha='right', fontsize=8, color='#95A5A6')
    ```

### B. Font Sizing
*   **Title**: 12pt (Bold)
*   **Axis Labels**: 10pt
*   **Ticks**: 8pt
*   **Legend**: 8pt

---

## 4. Chart Types & Best Practices

| Data Type | Chart Style | Rule |
|:----------|:------------|:-----|
| **Time Series** | Line Chart | Use `linewidth=1.5`. Fill area under curve ONLY if meaningful (e.g. Drawdown). |
| **Distributions** | KDE / Histogram | Use `alpha=0.1` fill. Add vertical line for Mean/VaR. |
| **Regimes** | Background Shade | Use `axvspan` with `alpha=0.15`. Green for Calm, Red for Stress. |
| **Correlation** | Heatmap | Use `cmap='RdBu_r'` (Red=Neg, Blue=Pos) centered at 0. |

### Example Recipe (Matplotlib)
```python
def setup_plot_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.facecolor'] = '#FFFFFF'
    plt.rcParams['text.color'] = '#2C3E50'
    plt.rcParams['axes.labelcolor'] = '#2C3E50'
    plt.rcParams['xtick.color'] = '#2C3E50'
    plt.rcParams['ytick.color'] = '#2C3E50'

def draw_regime_overlay(ax, dates, probs):
    """Shade background Red based on Stress Probability."""
    ax.fill_between(dates, 0, 1, where=probs>0.5, 
                    facecolor='#E74C3C', alpha=0.15, transform=ax.get_xaxis_transform())
```
