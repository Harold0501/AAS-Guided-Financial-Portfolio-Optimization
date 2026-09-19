# Pre-model feature freeze

Frozen before any B/C selector experiment. Selection uses: valid contract, deterministic
computability, semantic recurrence across generations, nonredundancy, and distinction between
B and C. It does not use algorithm outcomes or selector performance.

## B consensus

`volume_return_correlation`, `cross_sectional_return_skewness`, `negative_return_persistence`,
`volume_volatility_ratio`, `daily_price_range`, `return_entropy`, `tail_return_co_occurrence`,
`volatility_concentration`, `return_serial_correlation`, `missingness_volatility_correlation`.

## C complementary consensus

`correlation_stability`, `cross_sectional_return_kurtosis`, `return_volume_joint_skewness`,
`return_volume_joint_kurtosis`, `extreme_return_participation`, `volume_adjusted_dispersion`,
`skew_volatility_ratio`, `return_mad_ratio`, `correlation_duration`,
`cross_asset_return_asymmetry`.

Exact formulas and edge policies are encoded in `feature_engine.py`. All are backward-looking,
asset-order invariant, and computed by Python rather than directly valued by the LLM.
