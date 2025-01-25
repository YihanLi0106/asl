"""
Script Name: Interest Rate Data Visualization

Author: Evelyn Li

Description:
This script processes and visualizes interest rate data from two datasets:
1. `FRBNY Rates.xlsx` - Contains Federal Reserve Bank of New York interest rate data.
2. `Repo GC Rates.xlsx` - Contains general collateral repo rate data.

The script performs the following steps:
1. Load the data from the provided Excel files.
2. Clean and preprocess the data, including removing duplicates and converting date formats.
3. Transform the `FRBNY Rates` dataset to analyze interest rates by rate type.
4. Merge the transformed data with the repo rates data.
5. Calculate and visualize the spread between interest rates, rolling averages, and rolling correlations.
6. Visualize the relationship between different interest rate types over time.

"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data and Data Processing
try:
    frbny_rates = pd.read_excel('FRBNY Rates.xlsx')
    repo_gc_rates = pd.read_excel('Repo GC Rates.xlsx')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"Unexpected error: {e}")
    exit()

# Display dataset information for verification
print("FRBNY Rates Data Info:")
print(frbny_rates.info())
print("\nRepo GC Rates Data Info:")
print(repo_gc_rates.info())

# Remove duplicates
frbny_rates.drop_duplicates(inplace=True)
repo_gc_rates.drop_duplicates(inplace=True)



# 2. Convert Date Columns to Datetime Format and Handle errors
frbny_rates['Effective Date'] = pd.to_datetime(frbny_rates['Effective Date'], errors='coerce') 
repo_gc_rates['Trade Date'] = pd.to_datetime(repo_gc_rates['Trade Date'], format='%Y%m%d', errors='coerce') 
frbny_rates.dropna(subset=['Effective Date'], inplace=True)
repo_gc_rates.dropna(subset=['Trade Date'], inplace=True)



# 3. Reshape the FRBNY Rates DataFrame to Analyze Rates by Type
rate_df = frbny_rates.pivot(index='Effective Date', columns='Rate Type', values='Rate (%)') 



# 4. Merge with Repo GC Rates
# Align repo rates with the reshaped FRBNY rates by date
repo_gc_rates.set_index('Trade Date', inplace=True)
rate_df = pd.merge(rate_df, repo_gc_rates['General Collateral Rate'], how='inner', left_index=True, right_index=True)


# 5. Descriptive Statistics and Technical Indicators
print('Descriptive Statistics of Rate:')
print(rate_df.describe())
print('\nCorrelation Matrix of Rate:')
print(rate_df.corr())

technical_indicators = pd.DataFrame(index=rate_df.index)
# Calculate the Spread between the rates
technical_indicators['Spread_EFFR_SOFR'] = rate_df['EFFR'] - rate_df['SOFR']
technical_indicators['Spread_EFFR_GCR'] = rate_df['EFFR'] - rate_df['General Collateral Rate']
technical_indicators['Spread_SOFR_GCR'] = rate_df['SOFR'] - rate_df['General Collateral Rate']

# Calculate the 30-day Moving Averages
technical_indicators['EFFR_MA'] = rate_df['EFFR'].rolling(window=30).mean()
technical_indicators['SOFR_MA'] = rate_df['SOFR'].rolling(window=30).mean()
technical_indicators['GCR_MA'] = rate_df['General Collateral Rate'].rolling(window=30).mean()

# Calculate the 30-day Rolling Correlations
technical_indicators['Corr_EFFR_SOFR'] = rate_df['EFFR'].rolling(window=30).corr(rate_df['SOFR'])
technical_indicators['Corr_EFFR_GCR'] = rate_df['EFFR'].rolling(window=30).corr(rate_df['General Collateral Rate'])
technical_indicators['Corr_SOFR_GCR'] = rate_df['SOFR'].rolling(window=30).corr(rate_df['General Collateral Rate'])
""" Infinite values are observed in the rolling correlations between EFFR and SOFR, as well as EFFR and GCR.
    This occurs because EFFR remains constant for extended periods, resulting in zero variance
    and making the correlation calculation undefined during those windows.
    To handle this issue, infinite values are replaced with None to exclude them from the visualization."""
technical_indicators.replace([float('inf'), float('-inf')], None, inplace=True)



# 6. Data Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# (a). Line Plot of Rates Over Time
rate_df.plot(ax=axes[0, 0])
axes[0, 0].set_title('Daily Rates Over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Rates (%)')
axes[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 0))
axes[0, 0].grid(True)
axes[0, 0].tick_params(axis='x', rotation=30)

# (b). Spread Analysis
axes[0, 1].plot(technical_indicators.index, technical_indicators['Spread_EFFR_SOFR'], label='EFFR vs SOFR')
axes[0, 1].plot(technical_indicators.index, technical_indicators['Spread_EFFR_GCR'], label='EFFR vs General Collateral Rate')
axes[0, 1].plot(technical_indicators.index, technical_indicators['Spread_SOFR_GCR'], label='SOFR vs General Collateral Rate')
axes[0, 1].set_title('Rate Spreads Over Time')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Spread (%)')
axes[0, 1].legend(loc='lower left', bbox_to_anchor=(0, 0))
axes[0, 1].grid(True)
axes[0, 1].tick_params(axis='x', rotation=30)

# (c). Rolling Averages
technical_indicators[['EFFR_MA', 'SOFR_MA', 'GCR_MA']].plot(ax=axes[1, 0])
axes[1, 0].set_title('30-Day Rolling Averages')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Rates (%)')
axes[1, 0].legend(loc='lower left', bbox_to_anchor=(0, 0))
axes[1, 0].grid(True)
axes[1, 0].tick_params(axis='x', rotation=30)

# (d). Rolling Correlations
axes[1, 1].plot(technical_indicators.index, technical_indicators['Corr_EFFR_SOFR'], label='Corr: EFFR vs SOFR')
axes[1, 1].plot(technical_indicators.index, technical_indicators['Corr_EFFR_GCR'], label='Corr: EFFR vs General Collateral Rate')
axes[1, 1].plot(technical_indicators.index, technical_indicators['Corr_SOFR_GCR'], label='Corr: SOFR vs General Collateral Rate')
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=0.8)
axes[1, 1].set_title('30-Day Rolling Correlations Between Rates')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Correlation')
axes[1, 1].legend(loc='lower left', bbox_to_anchor=(0, 0))
axes[1, 1].grid(True)
axes[1, 1].tick_params(axis='x', rotation=30)

fig.suptitle('The Relationship between the Rates', fontsize=14, y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('EvelynLi-Interest_Rate_Data_Visualization.png')
plt.show()
