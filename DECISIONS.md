# Modeling decisions

This file records choices that affect the credit-risk estimates. Data files are
not tracked in Git. The model-ready handoff was extracted from
`~/Downloads/data_files.zip`.

- **Data source:** Freddie Mac SFLLD Release 48, Standard full annual archives
  for 2008, 2018 and 2023, all four quarters in each year. The three selected
  years provide a crisis period, a later development period and an out-of-time
  test period. They are not a representative sample of all Freddie Mac loans.
- **Prediction point:** first observed monthly performance record with current
  delinquency status, positive UPB, no termination and no detected value
  violation. This is an acquisition/servicing snapshot, not an application-time
  decision.
- **Target:** serious delinquency in calendar months t+1 through t+12, where
  serious means numeric delinquency of at least 3 months or REO acquisition.
  The target is 0 only after 12 distinct months with known status and no event.
  Censored and incomplete histories remain unlabeled.
- **Splits:** 2008 for training, H1 2018 for validation, remaining mature 2018
  observations for calibration, and 2023 for final testing. The test set is not
  used for fitting, early stopping, feature selection or calibrator selection.
- **Refinance leakage:** later loans with a disclosed pre-HARP predecessor in a
  different split are excluded. Undisclosed repeat borrowers remain a residual
  limitation.
- **COVID treatment:** COVID is absent from the 2023 test outcome window. It can
  affect 2018 calibration observations. Borrower-assistance fields are not used
  as predictors, and the 90-day-delinquency proxy remains unchanged. This is
  reported as a calibration limitation rather than applied inconsistently.
- **PD models:** L2 logistic regression and XGBoost. Preprocessing, category
  levels and tuning are fit on training data; validation chooses XGBoost
  stopping and calibration method. Calibration candidates are fit on the early
  portion of the calibration split and selected on its later portion, then the
  chosen calibrator is refit on the full calibration split.
- **Macro variables:** the supplied handoff has no macro series. The PD models
  therefore contain loan and servicing-snapshot variables only. Stress testing
  uses explicit PD multipliers and recovery haircuts, not claimed macro paths.
- **LGD:** only resolved credit dispositions with positive observed EAD and
  disclosed loss are eligible. Severity is fit on development splits, never on
  the test split. Current LTV is a static proxy using observed EAD and original
  property value because no state HPI series was supplied. Predicted LGD is
  clipped to [0,1] only for ECL arithmetic; raw observed fractions are retained
  for diagnostics.
- **ECL:** 12-month expected credit loss, `PD * LGD * EAD`, without discounting
  or lifetime staging. EAD is the snapshot UPB. This is not CECL or IFRS 9
  lifetime ECL.
- **Stress:** baseline, adverse and severe scenarios multiply PD and increase
  LGD through recovery haircuts. Scenario probabilities are assumptions, not
  fitted forecasts.
- **Source validation:** the nested 2008 sample has 31 origination and 35
  performance fields. All 2,459,811 performance rows were width-checked.
  Among 02/03/09 resolutions, 96.97% of disclosed actual losses are positive,
  so Release 48 losses are not flipped. This differs from the old negative-sign
  legacy assumption in the earlier plan.
- **Out-of-time calibration finding:** calibrated XGBoost predicts a 1.219%
  mean 12-month PD on 2023 test loans versus a 0.637% observed event rate
  (`pred_over_actual = 1.91`). The ranking remains useful (AUC 0.722), but the
  level is not used without this warning. A newer-period recalibration or
  explicit PD haircut is required before portfolio decisioning.
- **LGD sign finding:** the static current-LTV proxy has a negative univariate
  correlation (-0.163) and a negative conditional Ridge coefficient. Original
  LTV is also negatively correlated (-0.166). The model preserves the observed
  relationship rather than forcing the expected positive sign; selected
  resolution and missing HPI data remain limitations.
- **Back-test limitation:** aggregate expected defaults are 1.91 times observed
  test defaults. Only 328 test defaults have resolved disclosed losses, so the
  loss comparison is incomplete and is not presented as a cash-flow validation.
