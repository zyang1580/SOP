import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats
from kneed import KneeLocator

# ---------------------------------------------------------
# UTILITIES & GRANGER CAUSALITY (Retained from previous step)
# ---------------------------------------------------------
def timeit_decorator(func):
    """Decorator to log execution time for robust performance tracking."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[{func.__name__}] Execution Time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def recursive_bootstrap_residuals(residuals, block_size, B_iterations, random_state=None):
    """Generates recursive block bootstrapped residuals to address serial correlation (for Granger)."""
    if random_state is not None:
        np.random.seed(random_state)
    n = len(residuals)
    block_size = max(1, min(block_size, n))
    num_blocks = n - block_size + 1
    if num_blocks <= 0:
        block_size = max(1, n // 3)
        num_blocks = n - block_size + 1
        
    blocks = np.lib.stride_tricks.sliding_window_view(residuals, block_size)
    num_needed = (n + block_size - 1) // block_size
    block_indices = np.random.choice(num_blocks, size=(B_iterations, num_needed), replace=True)
    
    boot_resids = []
    for i in range(B_iterations):
        selected_blocks = blocks[block_indices[i]]
        sample = selected_blocks.flatten()[:n]
        boot_resids.append(sample)
    return boot_resids

def create_lagged_data(target_exposure, focal_supply, control_supply=None, max_lags=6, exog_vars=None):
    """Constructs the lagged panel data architecture for dynamic regression."""
    data_dict = {'target_exposure': target_exposure, 'focal_supply': focal_supply}
    if control_supply is not None:
        data_dict['control_supply'] = control_supply
    data = pd.DataFrame(data_dict)
    
    if exog_vars is not None:
        if isinstance(exog_vars, pd.DataFrame):
            data = pd.concat([data, exog_vars], axis=1)
        else:
            data = pd.concat([data, pd.DataFrame(exog_vars)], axis=1)
            
    for col in ['target_exposure', 'focal_supply', 'control_supply']:
        if col in data.columns:
            for lag in range(1, max_lags + 1):
                data[f'{col}_lag{lag}'] = data[col].shift(lag)
    data = data.dropna()
    
    y_series = data['target_exposure']
    
    X_unrestricted = pd.DataFrame()
    for lag in range(1, max_lags + 1):
        X_unrestricted[f'target_exposure_lag{lag}'] = data[f'target_exposure_lag{lag}']
        if control_supply is not None and f'control_supply_lag{lag}' in data.columns:
            X_unrestricted[f'control_supply_lag{lag}'] = data[f'control_supply_lag{lag}']
        X_unrestricted[f'focal_supply_lag{lag}'] = data[f'focal_supply_lag{lag}']
        
    if exog_vars is not None:
        exog_cols = exog_vars.columns.tolist() if isinstance(exog_vars, pd.DataFrame) else [f'exog_{i}' for i in range(exog_vars.shape[1])]
        for col in exog_cols:
            if col in data.columns:
                X_unrestricted[col] = data[col]
                
    X_restricted = pd.DataFrame()
    for lag in range(1, max_lags + 1):
        X_restricted[f'target_exposure_lag{lag}'] = data[f'target_exposure_lag{lag}']
        if control_supply is not None and f'control_supply_lag{lag}' in data.columns:
            X_restricted[f'control_supply_lag{lag}'] = data[f'control_supply_lag{lag}']
            
    if exog_vars is not None:
        for col in exog_cols:
            if col in data.columns:
                X_restricted[col] = data[col]
                
    X_unrestricted = sm.add_constant(X_unrestricted, has_constant='add')
    X_restricted = sm.add_constant(X_restricted, has_constant='add')
    
    return X_unrestricted, X_restricted, y_series, data

@timeit_decorator
def bootstrap_granger_test(target_exposure, focal_supply, control_supply=None, max_lags=6, exog_vars=None, block_size=None, B_iterations=999, random_state=None):
    """Performs the Granger causality test reinforced with block bootstrapping."""
    if random_state is not None:
        np.random.seed(random_state)
    
    X_unrestricted, X_restricted, y_lagged, full_data = create_lagged_data(target_exposure, focal_supply, control_supply, max_lags, exog_vars)
    n = len(y_lagged)
    if block_size is None:
        block_size = int(n ** (1/3))
        
    model_unrestricted = OLS(y_lagged, X_unrestricted)
    results_unrestricted = model_unrestricted.fit()
    rss_u = results_unrestricted.ssr
    df_u = results_unrestricted.df_resid
    
    model_restricted = OLS(y_lagged, X_restricted)
    results_restricted = model_restricted.fit()
    rss_r = results_restricted.ssr
    
    q = X_unrestricted.shape[1] - X_restricted.shape[1]
    F_original = ((rss_r - rss_u) / q) / (rss_u / df_u)
    p_F = 1 - stats.f.cdf(F_original, q, df_u)
    
    resid_constrained = results_restricted.resid.values
    boot_resids = recursive_bootstrap_residuals(resid_constrained, block_size, B_iterations)
    
    boot_F_stats = []
    beta_r = results_restricted.params.values
    X_restricted_values = X_restricted.values
    
    y_lag_cols = [col for col in X_restricted.columns if col.startswith('target_exposure_lag')]
    y_lag_indices = [list(X_restricted.columns).index(col) for col in y_lag_cols]
    
    for b in range(B_iterations):
        resid_boot = boot_resids[b]
        y_boot = np.zeros(n)
        y_boot[:max_lags] = y_lagged.values[:max_lags]
        
        for t in range(max_lags, n):
            features = np.ones(len(X_restricted.columns))
            for lag_idx, lag in enumerate(range(1, max_lags + 1)):
                if t - lag >= 0:
                    features[y_lag_indices[lag_idx]] = y_boot[t - lag]
            for col_idx, col in enumerate(X_restricted.columns):
                if col not in y_lag_cols + ['const']:
                    features[col_idx] = X_restricted_values[t, col_idx]
            y_boot[t] = np.dot(features, beta_r) + resid_boot[t]
            
        y_boot_series = pd.Series(y_boot, index=y_lagged.index)
        try:
            results_u_boot = OLS(y_boot_series, X_unrestricted).fit()
            results_r_boot = OLS(y_boot_series, X_restricted).fit()
            F_boot = ((results_r_boot.ssr - results_u_boot.ssr) / q) / (results_u_boot.ssr / df_u)
            boot_F_stats.append(F_boot)
        except:
            continue
            
    boot_F_stats = np.array(boot_F_stats)
    p_boot = np.mean(boot_F_stats >= F_original) if len(boot_F_stats) > 0 else np.nan
    
    return {
        'F_original': F_original,
        'p_F': p_F,
        'p_boot': p_boot,
        'model_unrestricted': results_unrestricted
    }

def find_optimal_lag_kneedle(target_exposure, focal_supply, control_supply, exog_vars, max_lag=6):
    """Utilizes the Kneedle algorithm to identify the optimal structural lag."""
    lags, aic_values = [], []
    for lag in range(1, max_lag + 1):
        X_unrestricted, X_restricted, y_lagged, _ = create_lagged_data(target_exposure, focal_supply, control_supply, max_lags=lag, exog_vars=exog_vars)
        model = sm.OLS(y_lagged, X_unrestricted).fit()
        lags.append(lag)
        aic_values.append(model.aic)
        
    try:
        kneedle = KneeLocator(lags, aic_values, curve='convex', direction='decreasing', interp_method='polynomial')
        optimal_lag = kneedle.knee if kneedle.knee is not None else min(zip(lags, aic_values), key=lambda x: x[1])[0]
    except:
        optimal_lag = min(zip(lags, aic_values), key=lambda x: x[1])[0]
        
    return optimal_lag, lags, aic_values

# ---------------------------------------------------------
# DOLS & SHIN RESIDUAL STATIONARITY TEST (New Additions)
# ---------------------------------------------------------
class ShinResidualStationarityTest:
    """
    Shin (1994) Test for Residual Stationarity.
    Used to confirm that DOLS residuals are stationary (H0), thereby precluding spurious regression.
    """
    def __init__(self):
        self.critical_values_standard = {
            1: {0.01: 0.027, 0.05: 0.043, 0.10: 0.057},
            2: {0.01: 0.023, 0.05: 0.035, 0.10: 0.046},
            3: {0.01: 0.021, 0.05: 0.030, 0.10: 0.038},
            4: {0.01: 0.018, 0.05: 0.026, 0.10: 0.033},
            5: {0.01: 0.016, 0.05: 0.023, 0.10: 0.029}
        }
        self.critical_values_demeaned = {
            1: {0.01: 0.020, 0.05: 0.029, 0.10: 0.035},
            2: {0.01: 0.017, 0.05: 0.024, 0.10: 0.029},
            3: {0.01: 0.015, 0.05: 0.021, 0.10: 0.025},
            4: {0.01: 0.014, 0.05: 0.019, 0.10: 0.022},
            5: {0.01: 0.013, 0.05: 0.017, 0.10: 0.019}
        }
        
    def _compute_long_run_variance(self, residuals, lags=10):
        T = len(residuals)
        gamma = [np.mean(residuals**2)]
        for j in range(1, lags + 1):
            gamma.append(np.mean(residuals[j:] * residuals[:-j]))
        weights = [1 - j/(lags+1) for j in range(lags + 1)] # Bartlett kernel
        lrv = gamma[0] + 2 * sum(weights[j] * gamma[j] for j in range(1, lags + 1))
        return lrv
        
    def shin_test(self, residuals, num_regressors, trend='c', lags=10):
        T = len(residuals)
        S_t = np.cumsum(residuals)
        omega_hat = self._compute_long_run_variance(residuals, lags=lags)
        CI_stat = np.sum(S_t**2) / (T**2 * omega_hat)
        
        cv_dict = self.critical_values_demeaned if trend == 'c' else self.critical_values_standard
        critical_values = cv_dict.get(num_regressors, {})
        
        # Simplified p-value interpolation
        if not critical_values: p_value = np.nan
        else:
            sorted_alpha = sorted(critical_values.keys())
            sorted_cv = [critical_values[alpha] for alpha in sorted_alpha]
            if CI_stat <= sorted_cv[0]: p_value = sorted_alpha[0]
            elif CI_stat >= sorted_cv[-1]: p_value = sorted_alpha[-1]
            else:
                for i in range(len(sorted_alpha)-1):
                    if sorted_cv[i] <= CI_stat <= sorted_cv[i+1]:
                        p_value = np.exp(np.log(sorted_alpha[i]) + (np.log(sorted_alpha[i+1]) - np.log(sorted_alpha[i])) * (CI_stat - sorted_cv[i]) / (sorted_cv[i+1] - sorted_cv[i]))
                        break

        conclusion = "Fail to reject H0 - Residuals are stationary (No spurious regression)" if p_value >= 0.05 else "Reject H0 - Spurious regression risk"
        
        return {
            'test_statistic': CI_stat,
            'p_value': p_value,
            'conclusion': conclusion,
            'H0': 'Residuals are stationary',
            'H1': 'Residuals have a unit root'
        }

def estimate_dols(y, X_base, p, supply_scales):
    """Estimates the Dynamic OLS (DOLS) model incorporating leads and lags."""
    X_dols = X_base.copy()
    for col in supply_scales:
        for lag in range(-p, p+1):
            if lag != 0:
                diff_col = f'delta_{col}_lag{lag}' if lag > 0 else f'delta_{col}_lead{abs(lag)}'
                X_dols[diff_col] = X_base[col].diff().shift(-lag)
    
    valid_mask = ~X_dols.isna().any(axis=1)
    X_dols_clean = X_dols[valid_mask]
    y_clean = y[valid_mask]
    model = sm.OLS(y_clean, X_dols_clean).fit(cov_type='HAC', cov_kwds={"maxlags": 5})
    return model, X_dols_clean, y_clean

def select_dols_order_aic(y, X_base, max_p, supply_scales):
    """Selects optimal DOLS lead/lag order based on AIC."""
    aic_values, models = [], []
    for p in range(0, max_p + 1):
        try:
            model, X_clean, y_clean = estimate_dols(y, X_base, p, supply_scales)
            n, k = len(y_clean), X_clean.shape[1]
            aic = 2 * k + n * np.log(model.ssr / n)
            aic_values.append(aic)
            models.append(model)
        except:
            aic_values.append(np.inf)
            models.append(None)
    best_p_aic = np.argmin(aic_values)
    return best_p_aic, models[best_p_aic]

def block_bootstrap_residuals_dols(residuals, block_size, B, random_state=None):
    """Simple moving block bootstrap for DOLS residuals (no y-lags)."""
    if random_state is not None: np.random.seed(random_state)
    n = len(residuals)
    block_size = max(1, min(block_size if block_size else int(n**(1/3)), n))
    num_blocks = n - block_size + 1
    
    if num_blocks > 0:
        blocks = np.array([residuals[i:i + block_size] for i in range(num_blocks)])
        num_needed = (n + block_size - 1) // block_size
        boot_resids = []
        for _ in range(B):
            block_idx = np.random.randint(0, num_blocks, size=num_needed)
            boot_sample = np.concatenate(blocks[block_idx])[:n]
            boot_resids.append(boot_sample)
    else:
        boot_resids = [np.random.choice(residuals, size=n, replace=True) for _ in range(B)]
    return boot_resids

@timeit_decorator
def bootstrap_dols_params(model, X_clean, y_clean, supply_scales, B=1999, block_size=None, alpha=0.05, random_state=42):
    """Calculates bootstrapped confidence intervals for DOLS parameters."""
    if random_state is not None: np.random.seed(random_state)
    n = len(y_clean)
    residuals, fitted_values = model.resid.values, model.fittedvalues.values
    boot_resids = block_bootstrap_residuals_dols(residuals, block_size, B, random_state)
    
    param_names = model.model.exog_names
    main_indices = [param_names.index(var) for var in supply_scales if var in param_names]
    boot_coefs = np.zeros((B, len(main_indices)))
    
    valid_count = 0
    for b in range(B):
        y_boot = fitted_values + boot_resids[b]
        try:
            model_boot = sm.OLS(y_boot, X_clean).fit()
            boot_coefs[valid_count, :] = model_boot.params.values[main_indices]
            valid_count += 1
        except: continue
        
    boot_coefs = boot_coefs[:valid_count, :]
    results, original_coefs = {}, model.params.values[main_indices]
    
    for i, var in enumerate(supply_scales):
        if var in param_names:
            coefs_boot = boot_coefs[:, i]
            se_boot = np.std(coefs_boot, ddof=1)
            t_stat = original_coefs[i] / se_boot if se_boot > 0 else np.nan
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - X_clean.shape[1])) if not np.isnan(t_stat) else np.nan
            ci_percentile = np.percentile(coefs_boot, [100*alpha/2, 100*(1-alpha/2)])
            
            try:
                z0 = stats.norm.ppf(np.mean(coefs_boot < original_coefs[i]))
                ci_bca = np.percentile(coefs_boot, [100*stats.norm.cdf(2*z0 + stats.norm.ppf(alpha/2)), 100*stats.norm.cdf(2*z0 + stats.norm.ppf(1-alpha/2))])
            except:
                ci_bca = [np.nan, np.nan]
                
            results[var] = {
                'original_coef': original_coefs[i], 'boot_se': se_boot, 't_stat': t_stat,
                'p_value': p_value, 'ci_percentile': ci_percentile, 'ci_bca': ci_bca
            }
    return results

def enhanced_dols_analysis(y, X_base, max_lags=4, supply_scales=None, do_bootstrap=True, bootstrap_B=1999, random_state=42):
    """Orchestrates the DOLS estimation, Shin Test for residual stationarity, and Bootstrapping."""
    if supply_scales is None: supply_scales = ['scale_aigc', 'scale_hgc']
    num_regressors = len(supply_scales)
    
    best_p_aic, final_model = select_dols_order_aic(y, X_base, max_lags, supply_scales)
    _, X_final, y_final = estimate_dols(y, X_base, best_p_aic, supply_scales)
    
    # Shin Test for Residual Stationarity (Replaces ADF/LB/JB)
    shin_test = ShinResidualStationarityTest()
    shin_results = shin_test.shin_test(final_model.resid, num_regressors, trend='c', lags=10)
    
    bootstrap_results = None
    if do_bootstrap:
        bootstrap_results = bootstrap_dols_params(final_model, X_final, y_final, main_vars=supply_scales, B=bootstrap_B, random_state=random_state)
        
    # Calculate partial residuals for plotting
    partial_residuals = {}
    for target in supply_scales:
        if target in X_final.columns:
            others = [c for c in X_final.columns if c != target]
            X_others = X_final[others]
            res_y = sm.OLS(y_final, X_others).fit().resid
            res_x = sm.OLS(X_final[target], X_others).fit().resid
            partial_residuals[target] = {'resid_x': res_x.values, 'resid_y': res_y.values}
            
    return final_model, shin_results, bootstrap_results, partial_residuals
