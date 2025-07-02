import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class InvestmentAnalyzer:
    def __init__(self):
        self.monthly_contribution = 250  # euros
        
        # Asset allocation - Updated to include STOXX Europe 600 ETF
        self.sp500_allocation = 0.50  # 50%
        self.stoxx_div_allocation = 0.30  # 30%
        self.stoxx_eu600_allocation = 0.20  # 20% STOXX Europe 600 ETF
        
        # Enhanced scenarios with geopolitical and regional considerations
        self.scenarios = {
            'global_recession': {
                'sp500': -0.05,        # -5% annual return (severe bear market)
                'stoxx_div': -0.02,    # -2% annual return (defensive but still negative)
                'stoxx_eu600': -0.08,  # -8% annual return (European crisis)
                'description': 'Global recession with significant market downturns across all regions'
            },
            'us_slowdown': {
                'sp500': 0.02,         # 2% annual return (US economic slowdown)
                'stoxx_div': 0.06,     # 6% annual return (global dividends more resilient)
                'stoxx_eu600': 0.09,   # 9% annual return (Europe benefits from US weakness)
                'description': 'US economic slowdown while Europe remains relatively strong'
            },
            'eu_crisis': {
                'sp500': 0.12,         # 12% annual return (US benefits from flight to quality)
                'stoxx_div': 0.03,     # 3% annual return (dividend stocks defensive)
                'stoxx_eu600': -0.02,  # -2% annual return (European crisis)
                'description': 'European economic crisis with capital flight to US markets'
            },
            'pessimistic': {
                'sp500': 0.03,         # 3% annual return (prolonged bear market)
                'stoxx_div': 0.02,     # 2% annual return (dividend focus, defensive)
                'stoxx_eu600': 0.02,   # 2% annual return (European market stress)
                'description': 'Prolonged bear market with low growth across all regions'
            },
            'stagnation': {
                'sp500': 0.05,         # 5% annual return (economic stagnation)
                'stoxx_div': 0.04,     # 4% annual return (dividends provide some return)
                'stoxx_eu600': 0.04,   # 4% annual return (low growth environment)
                'description': 'Economic stagnation with minimal growth and inflation concerns'
            },
            'average': {
                'sp500': 0.10,         # 10% annual return (historical average)
                'stoxx_div': 0.07,     # 7% annual return (dividend-focused)
                'stoxx_eu600': 0.08,   # 8% annual return (European equity average)
                'description': 'Historical average returns with normal economic cycles'
            },
            'moderate_growth': {
                'sp500': 0.12,         # 12% annual return (above average growth)
                'stoxx_div': 0.09,     # 9% annual return (strong dividend growth)
                'stoxx_eu600': 0.10,   # 10% annual return (solid European performance)
                'description': 'Moderate economic growth with stable geopolitical environment'
            },
            'optimistic': {
                'sp500': 0.15,         # 15% annual return (bull market)
                'stoxx_div': 0.12,     # 12% annual return (strong dividend growth)
                'stoxx_eu600': 0.13,   # 13% annual return (strong European growth)
                'description': 'Strong bull market with technological innovation and growth'
            },
            'tech_boom': {
                'sp500': 0.18,         # 18% annual return (tech-driven boom)
                'stoxx_div': 0.10,     # 10% annual return (dividends lag in tech boom)
                'stoxx_eu600': 0.14,   # 14% annual return (Europe benefits from tech spillover)
                'description': 'Technology-driven boom similar to late 1990s or 2010s'
            }
        }
        
        self.time_horizons = [5, 10, 15, 20, 25, 30]  # years
        
        # Define scenario categories for better organization
        self.scenario_categories = {
            'Crisis Scenarios': ['global_recession', 'us_slowdown', 'eu_crisis'],
            'Conservative Scenarios': ['pessimistic', 'stagnation'],
            'Normal Scenarios': ['average', 'moderate_growth'],
            'Optimistic Scenarios': ['optimistic', 'tech_boom']
        }
    
    def calculate_portfolio_return(self, sp500_return, stoxx_div_return, stoxx_eu600_return):
        """Calculate weighted portfolio return"""
        return (self.sp500_allocation * sp500_return + 
                self.stoxx_div_allocation * stoxx_div_return + 
                self.stoxx_eu600_allocation * stoxx_eu600_return)
    
    def calculate_compound_growth(self, annual_return, years, monthly_contribution):
        """Calculate compound growth with monthly contributions"""
        monthly_return = annual_return / 12
        months = years * 12
        
        # Arrays to store monthly values
        total_invested = np.zeros(months + 1)
        portfolio_value = np.zeros(months + 1)
        
        for month in range(1, months + 1):
            # Add monthly contribution
            total_invested[month] = total_invested[month-1] + monthly_contribution
            
            # Calculate new portfolio value (previous value grows + new contribution)
            portfolio_value[month] = (portfolio_value[month-1] * (1 + monthly_return) + 
                                    monthly_contribution)
        
        return total_invested, portfolio_value
    
    def run_analysis(self):
        """Run complete investment analysis"""
        results = {}
        
        for scenario_name, returns in self.scenarios.items():
            portfolio_return = self.calculate_portfolio_return(
                returns['sp500'], returns['stoxx_div'], returns['stoxx_eu600']
            )
            
            scenario_results = {}
            for years in self.time_horizons:
                total_invested, portfolio_value = self.calculate_compound_growth(
                    portfolio_return, years, self.monthly_contribution
                )
                
                final_value = portfolio_value[-1]
                total_contributions = total_invested[-1]
                gains = final_value - total_contributions
                roi_percentage = (gains / total_contributions) * 100 if total_contributions > 0 else 0
                
                scenario_results[years] = {
                    'total_invested': total_contributions,
                    'final_value': final_value,
                    'gains': gains,
                    'roi_percentage': roi_percentage,
                    'monthly_data': {
                        'invested': total_invested,
                        'value': portfolio_value
                    }
                }
            
            results[scenario_name] = scenario_results
        
        return results
    
    def create_detailed_plots(self, results):
        """Create comprehensive visualization plots and save each as separate image"""
        
        # Create directory for individual plots
        import os
        plot_dir = "individual_plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # Plot 1: Portfolio Growth Over Time (30 years detailed) - Key scenarios
        plt.figure(figsize=(12, 8))
        years_30 = 30
        months_30 = np.arange(0, years_30 * 12 + 1)
        
        # Select key scenarios for main plot
        key_scenarios = ['global_recession', 'pessimistic', 'average', 'optimistic', 'tech_boom']
        colors = ['#8B0000', '#FF6B6B', '#4ECDC4', '#45B7D1', '#32CD32']
        
        for i, scenario in enumerate(key_scenarios):
            if scenario in results:
                data = results[scenario][years_30]['monthly_data']
                plt.plot(months_30/12, data['value'], linewidth=3, 
                        label=f'{scenario.replace("_", " ").title()}', color=colors[i])
        
        # Add total invested line
        invested_data = results['average'][years_30]['monthly_data']['invested']
        plt.plot(months_30/12, invested_data, '--', alpha=0.6, linewidth=2,
                label='Total Invested', color='gray')
        
        plt.title('Portfolio Growth Over 30 Years - Multiple Scenarios\n(Monthly Contributions: €250)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Years')
        plt.ylabel('Portfolio Value (€)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/01_portfolio_growth_30years.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Geopolitical Impact Analysis
        plt.figure(figsize=(12, 8))
        geo_scenarios = ['us_slowdown', 'eu_crisis', 'average', 'global_recession']
        years_analysis = 20
        
        scenario_names = []
        final_values = []
        colors_geo = ['#FF6B6B', '#FFA500', '#4ECDC4', '#8B0000']
        
        for scenario in geo_scenarios:
            if scenario in results:
                scenario_names.append(scenario.replace('_', ' ').title())
                final_values.append(results[scenario][years_analysis]['final_value'])
        
        bars = plt.bar(range(len(scenario_names)), final_values, color=colors_geo, alpha=0.8)
        plt.title(f'Geopolitical Impact on Portfolio Value\n(After {years_analysis} Years)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Scenario')
        plt.ylabel('Final Value (€)')
        plt.xticks(range(len(scenario_names)), scenario_names, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='plain', axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, final_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                    f'€{value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/02_geopolitical_impact.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Regional Performance Comparison
        plt.figure(figsize=(12, 8))
        regions = ['S&P 500 (US)', 'STOXX Div 100 (Global)', 'STOXX EU 600 (Europe)']
        
        # Compare different scenarios
        comparison_scenarios = ['us_slowdown', 'eu_crisis', 'average']
        x = np.arange(len(regions))
        width = 0.25
        
        for i, scenario in enumerate(comparison_scenarios):
            if scenario in results:
                returns = self.scenarios[scenario]
                values = [returns['sp500']*100, returns['stoxx_div']*100, returns['stoxx_eu600']*100]
                plt.bar(x + i*width, values, width, label=scenario.replace('_', ' ').title(), 
                       alpha=0.8)
        
        plt.title('Regional Performance by Scenario\n(Annual Returns %)', fontsize=14, fontweight='bold')
        plt.xlabel('Region/Asset')
        plt.ylabel('Annual Return (%)')
        plt.xticks(x + width, regions)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/03_regional_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 4: Crisis vs Normal Times Analysis
        plt.figure(figsize=(10, 8))
        crisis_scenarios = ['global_recession', 'us_slowdown', 'eu_crisis']
        normal_scenarios = ['average', 'moderate_growth', 'optimistic']
        
        years_test = 15
        crisis_values = [results[s][years_test]['final_value'] for s in crisis_scenarios if s in results]
        normal_values = [results[s][years_test]['final_value'] for s in normal_scenarios if s in results]
        
        plt.boxplot([crisis_values, normal_values], labels=['Crisis Scenarios', 'Normal Scenarios'])
        plt.title(f'Portfolio Values Distribution\n(After {years_test} Years)', fontsize=14, fontweight='bold')
        plt.ylabel('Final Value (€)')
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/04_crisis_vs_normal.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 5: ROI Heatmap by Scenario and Time
        plt.figure(figsize=(12, 10))
        
        # Create ROI matrix
        scenario_list = list(results.keys())
        roi_matrix = []
        
        for scenario in scenario_list:
            roi_row = [results[scenario][years]['roi_percentage'] for years in self.time_horizons]
            roi_matrix.append(roi_row)
        
        roi_df = pd.DataFrame(roi_matrix, 
                             index=[s.replace('_', ' ').title() for s in scenario_list],
                             columns=[f'{y}Y' for y in self.time_horizons])
        
        sns.heatmap(roi_df, annot=True, fmt='.0f', cmap='RdYlGn', center=100, 
                   cbar_kws={'label': 'ROI (%)'})
        plt.title('ROI Heatmap by Scenario and Time Horizon', fontsize=14, fontweight='bold')
        plt.xlabel('Investment Period')
        plt.ylabel('Scenario')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/05_roi_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 6: Scenario Probability Impact
        plt.figure(figsize=(10, 8))
        
        # Assign realistic probabilities to scenarios
        scenario_probabilities = {
            'global_recession': 0.05,    # 5% chance over any given decade
            'us_slowdown': 0.15,         # 15% chance
            'eu_crisis': 0.10,           # 10% chance
            'pessimistic': 0.20,         # 20% chance
            'stagnation': 0.15,          # 15% chance
            'average': 0.25,             # 25% chance (most likely)
            'moderate_growth': 0.15,     # 15% chance
            'optimistic': 0.08,          # 8% chance
            'tech_boom': 0.02            # 2% chance (rare)
        }
        
        years_prob = 20
        expected_values = []
        probabilities = []
        scenario_names_prob = []
        
        for scenario, prob in scenario_probabilities.items():
            if scenario in results:
                expected_values.append(results[scenario][years_prob]['final_value'] * prob)
                probabilities.append(prob)
                scenario_names_prob.append(scenario.replace('_', ' ').title())
        
        # Create probability-weighted analysis
        plt.pie(probabilities, labels=scenario_names_prob, autopct='%1.1f%%', startangle=90)
        plt.title('Scenario Probability Distribution\n(Estimated Likelihood)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/06_scenario_probabilities.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 7: Portfolio Resilience Analysis
        plt.figure(figsize=(12, 8))
        
        # Calculate portfolio resilience (how well it performs in bad scenarios vs good)
        bad_scenarios = ['global_recession', 'us_slowdown', 'eu_crisis', 'pessimistic']
        good_scenarios = ['average', 'moderate_growth', 'optimistic', 'tech_boom']
        
        resilience_years = [10, 20, 30]
        bad_performance = []
        good_performance = []
        
        for years in resilience_years:
            bad_avg = np.mean([results[s][years]['final_value'] for s in bad_scenarios if s in results])
            good_avg = np.mean([results[s][years]['final_value'] for s in good_scenarios if s in results])
            bad_performance.append(bad_avg)
            good_performance.append(good_avg)
        
        x = np.arange(len(resilience_years))
        width = 0.35
        
        plt.bar(x - width/2, bad_performance, width, label='Crisis Scenarios (Avg)', color='red', alpha=0.7)
        plt.bar(x + width/2, good_performance, width, label='Favorable Scenarios (Avg)', color='green', alpha=0.7)
        
        plt.title('Portfolio Resilience Analysis\n(Average Performance by Scenario Type)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Investment Period (Years)')
        plt.ylabel('Average Final Value (€)')
        plt.xticks(x, resilience_years)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/07_portfolio_resilience.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 8: Asset Allocation
        plt.figure(figsize=(10, 8))
        
        allocations = [self.sp500_allocation, self.stoxx_div_allocation, self.stoxx_eu600_allocation]
        labels = ['S&P 500 ETF (50%)', 'STOXX Global Dividend 100 (30%)', 'STOXX Europe 600 ETF (20%)']
        colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        plt.pie(allocations, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
        plt.title('Portfolio Asset Allocation\n(Diversified Global Equity Portfolio)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/08_asset_allocation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 9: Monte Carlo-style Multiple Scenario Overview
        plt.figure(figsize=(16, 10))
        
        years_monte = 25
        months_monte = np.arange(0, years_monte * 12 + 1)
        
        # Plot all scenarios with transparency
        for scenario in results.keys():
            data = results[scenario][years_monte]['monthly_data']
            alpha = 0.4 if scenario in ['global_recession', 'tech_boom'] else 0.7
            linewidth = 2 if scenario in ['average', 'pessimistic', 'optimistic'] else 1.5
            
            plt.plot(months_monte/12, data['value'], linewidth=linewidth, alpha=alpha,
                    label=scenario.replace('_', ' ').title())
        
        plt.title('All Scenarios Portfolio Growth Comparison (25 Years)\nRealistic Geopolitical and Economic Scenarios', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Years')
        plt.ylabel('Portfolio Value (€)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/09_all_scenarios_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 10: Final Values Comparison by Time Horizon
        plt.figure(figsize=(14, 8))
        scenarios = list(results.keys())
        time_horizons = self.time_horizons
        
        x = np.arange(len(time_horizons))
        width = 0.08
        colors_bar = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
        
        for i, scenario in enumerate(scenarios):
            values = [results[scenario][years]['final_value'] for years in time_horizons]
            plt.bar(x + i*width, values, width, label=scenario.replace('_', ' ').title(), 
                   alpha=0.8, color=colors_bar[i])
        
        plt.title('Final Portfolio Values by Time Horizon - All Scenarios', fontsize=14, fontweight='bold')
        plt.xlabel('Investment Period (Years)')
        plt.ylabel('Final Value (€)')
        plt.xticks(x + width * (len(scenarios)/2), time_horizons)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/10_final_values_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 11: Monthly Contribution Impact Analysis
        plt.figure(figsize=(12, 8))
        contribution_scenarios = [200, 250, 300, 400, 500]
        years_test = 20
        
        for i, scenario in enumerate(['pessimistic', 'average', 'optimistic']):
            returns = self.scenarios[scenario]
            portfolio_return = self.calculate_portfolio_return(
                returns['sp500'], returns['stoxx_div'], returns['stoxx_eu600']
            )
            
            final_values = []
            for contribution in contribution_scenarios:
                _, portfolio_value = self.calculate_compound_growth(
                    portfolio_return, years_test, contribution
                )
                final_values.append(portfolio_value[-1])
            
            plt.plot(contribution_scenarios, final_values, marker='o', linewidth=2.5,
                    label=f'{scenario.capitalize()} Scenario', markersize=8)
        
        plt.title(f'Impact of Monthly Contribution Amount\n(After {years_test} Years)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Monthly Contribution (€)')
        plt.ylabel('Final Portfolio Value (€)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/11_contribution_impact.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nAll individual plots saved to '{plot_dir}/' directory:")
        plot_files = [
            "01_portfolio_growth_30years.png",
            "02_geopolitical_impact.png", 
            "03_regional_performance.png",
            "04_crisis_vs_normal.png",
            "05_roi_heatmap.png",
            "06_scenario_probabilities.png",
            "07_portfolio_resilience.png",
            "08_asset_allocation.png",
            "09_all_scenarios_comparison.png",
            "10_final_values_comparison.png",
            "11_contribution_impact.png"
        ]
        
        for file in plot_files:
            print(f"• {file}")

    def print_summary_table(self, results):
        """Print detailed summary table with geopolitical scenarios"""
        print("\n" + "="*120)
        print("COMPREHENSIVE INVESTMENT ANALYSIS - GEOPOLITICAL & ECONOMIC SCENARIOS")
        print("="*120)
        print(f"Monthly Contribution: €{self.monthly_contribution}")
        print(f"Asset Allocation: {self.sp500_allocation*100:.0f}% S&P 500 ETF, "
              f"{self.stoxx_div_allocation*100:.0f}% STOXX Global Dividend 100, "
              f"{self.stoxx_eu600_allocation*100:.0f}% STOXX Europe 600 ETF")
        print("Portfolio Type: 100% Equity with Geographic Diversification")
        print("\n")
        
        # Group scenarios by category
        for category, scenario_list in self.scenario_categories.items():
            print(f"\n{'='*50} {category.upper()} {'='*50}")
            
            for scenario_name in scenario_list:
                if scenario_name in results:
                    scenario_data = results[scenario_name]
                    
                    print(f"\n{scenario_name.upper().replace('_', ' ')} SCENARIO")
                    print("-" * 70)
                    print(f"Description: {self.scenarios[scenario_name]['description']}")
                    
                    # Calculate scenario returns
                    returns = self.scenarios[scenario_name]
                    portfolio_return = self.calculate_portfolio_return(
                        returns['sp500'], returns['stoxx_div'], returns['stoxx_eu600']
                    )
                    
                    print(f"Expected Annual Portfolio Return: {portfolio_return*100:.1f}%")
                    print(f"Regional Returns: US: {returns['sp500']*100:.1f}%, "
                          f"Global Div: {returns['stoxx_div']*100:.1f}%, "
                          f"Europe: {returns['stoxx_eu600']*100:.1f}%")
                    
                    print("\nKey Time Horizons:")
                    print(f"{'Years':<6} {'Invested':<12} {'Final Value':<15} {'Total Gains':<15} {'ROI %':<10}")
                    print("-" * 65)
                    
                    key_years = [10, 20, 30]
                    for years in key_years:
                        if years in scenario_data:
                            data = scenario_data[years]
                            print(f"{years:<6} €{data['total_invested']:>9,.0f} "
                                  f"€{data['final_value']:>12,.0f} "
                                  f"€{data['gains']:>12,.0f} "
                                  f"{data['roi_percentage']:>7.1f}%")

    def create_geopolitical_risk_analysis(self, results):
        """Additional geopolitical risk analysis"""
        print("\n" + "="*120)
        print("GEOPOLITICAL RISK ANALYSIS")
        print("="*120)
        
        print("\nRegional Exposure Risk Assessment:")
        print(f"• US Market Exposure (S&P 500): {self.sp500_allocation*100:.0f}%")
        print(f"• European Market Exposure (STOXX EU 600): {self.stoxx_eu600_allocation*100:.0f}%")
        print(f"• Global Dividend Exposure (STOXX Div 100): {self.stoxx_div_allocation*100:.0f}%")
        
        print("\nScenario Impact Analysis (20-year investment):")
        print("-" * 80)
        
        base_case = results['average'][20]['final_value']
        
        risk_scenarios = {
            'global_recession': 'Global economic recession',
            'us_slowdown': 'US economic slowdown',
            'eu_crisis': 'European economic crisis',
            'stagnation': 'Prolonged economic stagnation'
        }
        
        for scenario, description in risk_scenarios.items():
            if scenario in results:
                scenario_value = results[scenario][20]['final_value']
                impact = ((scenario_value - base_case) / base_case) * 100
                
                print(f"• {description:.<40} {impact:>+6.1f}% impact (€{scenario_value:>8,.0f})")
        
        print(f"\nBase case (Average scenario): €{base_case:,.0f}")
        
        # Diversification benefit analysis
        print(f"\nDiversification Benefits:")
        print(f"• Geographic diversification reduces single-country risk")
        print(f"• Dividend-focused allocation provides defensive characteristics")
        print(f"• Multi-region exposure helps in regional crisis scenarios")
    
    def create_asset_performance_comparison(self, results):
        """Create additional analysis showing individual asset performance"""
        print("\n" + "="*110)
        print("INDIVIDUAL ASSET PERFORMANCE ANALYSIS")
        print("="*110)
        
        # Calculate individual asset performance for 20 years
        years = 20
        monthly_contrib = self.monthly_contribution
        
        print(f"\nIndividual Asset Performance (20 years, €{monthly_contrib}/month allocated):")
        print("-" * 80)
        
        for scenario_name, returns in self.scenarios.items():
            print(f"\n{scenario_name.upper().replace('_', ' ')} SCENARIO:")
            
            # S&P 500 ETF performance (50% allocation)
            sp500_contrib = monthly_contrib * self.sp500_allocation
            _, sp500_value = self.calculate_compound_growth(returns['sp500'], years, sp500_contrib)
            sp500_final = sp500_value[-1]
            sp500_invested = sp500_contrib * years * 12
            
            # STOXX Dividend 100 performance (30% allocation)
            stoxx_div_contrib = monthly_contrib * self.stoxx_div_allocation
            _, stoxx_div_value = self.calculate_compound_growth(returns['stoxx_div'], years, stoxx_div_contrib)
            stoxx_div_final = stoxx_div_value[-1]
            stoxx_div_invested = stoxx_div_contrib * years * 12
            
            # STOXX Europe 600 performance (20% allocation)
            stoxx_eu_contrib = monthly_contrib * self.stoxx_eu600_allocation
            _, stoxx_eu_value = self.calculate_compound_growth(returns['stoxx_eu600'], years, stoxx_eu_contrib)
            stoxx_eu_final = stoxx_eu_value[-1]
            stoxx_eu_invested = stoxx_eu_contrib * years * 12
            
            print(f"  S&P 500 ETF (50%):           €{sp500_final:>8,.0f} (invested: €{sp500_invested:>6,.0f})")
            print(f"  STOXX Global Div 100 (30%):  €{stoxx_div_final:>8,.0f} (invested: €{stoxx_div_invested:>6,.0f})")
            print(f"  STOXX Europe 600 (20%):      €{stoxx_eu_final:>8,.0f} (invested: €{stoxx_eu_invested:>6,.0f})")
            print(f"  Total Portfolio:              €{sp500_final + stoxx_div_final + stoxx_eu_final:>8,.0f}")

def main():
    """Main execution function"""
    print("Starting Enhanced Investment Analysis with Geopolitical Scenarios...")
    print("Calculating compound growth with realistic economic and political scenarios...")
    print("Portfolio: 50% S&P 500 ETF + 30% STOXX Global Dividend 100 + 20% STOXX Europe 600 ETF")
    
    analyzer = InvestmentAnalyzer()
    results = analyzer.run_analysis()
    
    # Print summary
    analyzer.print_summary_table(results)
    
    # Geopolitical risk analysis
    analyzer.create_geopolitical_risk_analysis(results)
    
    # Additional asset performance analysis
    analyzer.create_asset_performance_comparison(results)
    
    # Create plots
    print("\nGenerating detailed visualizations with geopolitical scenarios...")
    analyzer.create_detailed_plots(results)
    
    print("\nAnalysis complete! Check 'individual_plots/' directory for all separate chart images.")
    
    # Enhanced insights
    print("\n" + "="*120)
    print("KEY INSIGHTS - GEOPOLITICAL & ECONOMIC SCENARIO ANALYSIS")
    print("="*120)
    
    # Compare extreme scenarios
    if 'global_recession' in results and 'tech_boom' in results:
        worst_case = results['global_recession'][30]
        best_case = results['tech_boom'][30]
        avg_case = results['average'][30]
        
        print(f"\nAfter 30 years of investing €250/month - Extreme Scenario Analysis:")
        print(f"• Global Recession:    €{worst_case['final_value']:,.0f} "
              f"(€{worst_case['gains']:,.0f} gains, {worst_case['roi_percentage']:.1f}% ROI)")
        print(f"• Average Scenario:    €{avg_case['final_value']:,.0f} "
              f"(€{avg_case['gains']:,.0f} gains, {avg_case['roi_percentage']:.1f}% ROI)")
        print(f"• Tech Boom:           €{best_case['final_value']:,.0f} "
              f"(€{best_case['gains']:,.0f} gains, {best_case['roi_percentage']:.1f}% ROI)")
        
        print(f"\nRange of outcomes: €{best_case['final_value'] - worst_case['final_value']:,.0f}")
    
    # Regional risk insights
    if 'us_slowdown' in results and 'eu_crisis' in results:
        us_slow = results['us_slowdown'][20]
        eu_crisis = results['eu_crisis'][20]
        
        print(f"\nRegional Risk Analysis (20 years):")
        print(f"• US Slowdown Scenario: €{us_slow['final_value']:,.0f}")
        print(f"• EU Crisis Scenario:   €{eu_crisis['final_value']:,.0f}")
        print(f"• Geographic diversification helps mitigate single-region risks")
    
    print(f"\nInvestment Strategy Recommendations:")
    print(f"• Maintain geographic diversification to reduce country-specific risks")
    print(f"• Consider rebalancing during extreme market conditions")
    print(f"• Dollar-cost averaging helps smooth out geopolitical volatility")
    print(f"• Long-term perspective is crucial during crisis periods")
    print(f"• Dividend-focused allocation provides some downside protection")

if __name__ == "__main__":
    main()
