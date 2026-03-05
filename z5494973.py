#!/usr/bin/env python3

import time
total_start_time = time.time()
from sys import argv, exit
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import KFold

"""
Credit Card Fraud Detection - Classification Task
Student ID: z5494973

Usage vlab:
    ssh
    mkdir -p comp9321/assign3
    cd comp9321/assign3
    python3 -m venv .venv
    source .venv/bin/activate

    pip install pandas
    pip install numpy
    pip install scikit-learn
    pip install lightgbm

    python3 z5494973.py train.csv test.csv

Target:
    - Regression RMSE <= 140                (8.52 away)
    - Classification F1 >= 0.97             (0.0199 away)
    - Combined Training Time < 120 seconds  (113.57s)

Key Design:
    - No merging of train and test datasets as per spec
    - Statistics saved from train, applied to test (avoid leakage)
    - Preprocessing function is dataset-agnostic
"""

#* =============================================================================
#* Part II: Regression - LightGBM
#* =============================================================================
class RegressionAmountPredictionPipeline():
    """LightGBM Regression Pipeline."""

    def __init__(self, student_id):
        self.start_time = time.time()
        self._n_splits = 5
        self._random_state = 42

        self.student_id = student_id
        self.model = None
        self.feature_columns = None

        #  Statistics
        self._city_pop_mean = None
        self._global_lat_mean = None
        self._global_long_mean = None
        self._global_merch_lat_mean = None
        self._global_merch_long_mean = None

        self._category_mean = None
        self._category_std = None
        self._category_count = None
        self._merchant_count = None

        self._global_amt_mean = None
        self._global_amt_std = None

        self._cust_mean_amt_full = None
        self._travel_month_mean = None

    # ==========================================================================
    def _fit_oof_cust_mean(self, df):
        kf = KFold(n_splits=self._n_splits, shuffle=True, random_state=self._random_state)
        oof = pd.Series(index=df.index, dtype=float)
        g = float(df["amt"].mean())
        for tr_idx, va_idx in kf.split(df):
            tr = df.iloc[tr_idx]
            fold_map = tr.groupby("cc_num")["amt"].mean()
            oof.iloc[va_idx] = df.iloc[va_idx]["cc_num"].map(fold_map).fillna(g).values
        self._cust_mean_amt_full = df.groupby("cc_num")["amt"].mean().to_dict()
        return oof

    # ==========================================================================
    def preprocess(self, df, is_train=True):
        df = df.copy()

        # ----------------------------------------------------------------------
        # Datetimes
        # ----------------------------------------------------------------------
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        s = df['trans_date_trans_time']
        df['trans_hour'] = s.dt.hour.astype('float')
        df['trans_month'] = s.dt.month.astype('float')
        df['trans_year'] = s.dt.year
        df['is_quarter'] = df['trans_month'].isin([3, 6, 9, 12]).astype(int)
        if s.notna().any():
            epoch_sec = (s.view('int64') // 10**9).astype('float')
            med_epoch = float(np.nanmedian(epoch_sec))
            df['unix_time'] = epoch_sec.fillna(med_epoch)
        else:
            df['unix_time'] = 0.0

        # ----------------------------------------------------------------------
        # Demographics
        # ----------------------------------------------------------------------
        df['gender_num'] = df['gender'].map({'F': 0, 'M': 1}).astype('float')
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = ((s - df['dob']).dt.total_seconds() / (365.25 * 24 * 3600)).clip(lower=0)
        df['age_bin'] = pd.cut(
            df['age'],
            bins=[18, 25, 35, 50, 65, 100],
            labels=['18-24', '25-34', '35-49', '50-64', '65+'],
            right=False, #edges bins goes into the RHS
            include_lowest=True #only affects the first bin
        ).astype('category').cat.codes

        # ----------------------------------------------------------------------
        # Travel feature
        # ----------------------------------------------------------------------
        if is_train:
            self._travel_month_mean = (
                df[df['category'] == 'travel']
                .groupby('trans_month')['amt']
                .mean()
                .to_dict()
            )
        df['travel_month_mean'] = df['trans_month'].map(self._travel_month_mean or {})

        # ----------------------------------------------------------------------
        # Location features + imputations for needed columns
        # ----------------------------------------------------------------------
        if is_train:
            self._global_lat_mean = float(df['lat'].mean())
            self._global_long_mean = float(df['long'].mean())
            self._global_merch_lat_mean = float(df['merch_lat'].mean())
            self._global_merch_long_mean = float(df['merch_long'].mean())
            self._city_pop_mean = float(df['city_pop'].mean())
        df['lat'] = df['lat'].fillna(self._global_lat_mean)
        df['long'] = df['long'].fillna(self._global_long_mean)
        df['merch_lat'] = df['merch_lat'].fillna(self._global_merch_lat_mean)
        df['merch_long'] = df['merch_long'].fillna(self._global_merch_long_mean)
        df['city_pop'] = df['city_pop'].fillna(self._city_pop_mean)

        # ----------------------------------------------------------------------
        # Distance (temp) + features derived from it
        # ----------------------------------------------------------------------
        _distance = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
        df['distance_bin'] = pd.cut(
            _distance,
            bins=[0, 1, 10, 50, 200, 1000, 5000, np.inf],
            labels=['<1km', '1-10km', '10-50km', '50-200km', '200-1k', '1k-5k', '5k+'],
            right=False
        ).astype('category').cat.codes
        df['dist_age_ratio'] = _distance / (df['age'] + 1)
        del _distance  # remove temp

        # ----------------------------------------------------------------------
        # Category code + stats used by selected features
        # ----------------------------------------------------------------------
        df['category_code'] = df['category'].astype('category').cat.codes.replace(-1, np.nan).astype('float')
        if is_train:
            self._category_count = df['category'].value_counts(dropna=False).to_dict()
            cat_stats = df.groupby('category')['amt'].agg(['mean', 'std'])
            self._category_mean = cat_stats['mean'].to_dict()
            self._category_std = cat_stats['std'].to_dict()
            self._global_amt_mean = float(df['amt'].mean())
            std_val = df['amt'].std()
            self._global_amt_std = float(std_val) if std_val is not None else 10.0

        g_mean = self._global_amt_mean if self._global_amt_mean is not None else 70.0
        g_std = self._global_amt_std if self._global_amt_std is not None else 10.0
        df['category_mean_amt'] = df['category'].map(self._category_mean or {}).fillna(g_mean)
        df['category_std_amt'] = df['category'].map(self._category_std or {}).fillna(g_std).replace(0, g_std)
        # ----------------------------------------------------------------------
        # Merchant counts (for merchant_cat_scale)
        # ----------------------------------------------------------------------
        if is_train:
            self._merchant_count = df['merchant'].value_counts(dropna=False).to_dict()
        df['merchant_txn_count'] = df['merchant'].map(self._merchant_count or {}).fillna(0).astype('float')

        # ----------------------------------------------------------------------
        # Job/state/merchant codes
        # ----------------------------------------------------------------------
        df['job_code'] = df['job'].astype('category').cat.codes.replace(-1, np.nan).astype('float')
        df['state_code'] = df['state'].astype('category').cat.codes.replace(-1, np.nan).astype('float')
        df['merchant_code'] = df['merchant'].astype('category').cat.codes.replace(-1, np.nan).astype('float')

        # OOF customer mean
        if is_train:
            df['cust_mean_amt'] = self._fit_oof_cust_mean(df)
        else:
            g = self._global_amt_mean if self._global_amt_mean is not None else 0.0
            df['cust_mean_amt'] = df['cc_num'].map(self._cust_mean_amt_full or {}).fillna(g).astype(float)

        # ----------------------------------------------------------------------
        # Merchant–category scale
        # ----------------------------------------------------------------------
        cat_cnt = df['category'].map(self._category_count or {}).fillna(0).astype(float)
        df['merchant_cat_scale'] = df['merchant_txn_count'] * cat_cnt
        return df

    # ==========================================================================
    def select_features(self, df):
        feature_columns = [
            'trans_hour',  'trans_month', 'trans_year',
            'is_quarter',
            'travel_month_mean',
            'unix_time',

            'lat', 'long',
            'merch_lat','merch_long',

            'city_pop',
            'age',
            'age_bin',
            'distance_bin',
            'dist_age_ratio',

            'category_code',
            'job_code',
            'merchant_code',
            'state_code',
            'gender_num',

            'category_mean_amt',
            'category_std_amt',
            'merchant_cat_scale',
            'cust_mean_amt',

        ]
        return df[feature_columns], feature_columns

    # ==========================================================================
    def train(self, train_df):
        train_processed = self.preprocess(train_df, is_train=True)
        X_train, self.feature_columns = self.select_features(train_processed)
        y_train = train_processed['amt']
        self.model = LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',          # 'gbdt', 'dart', 'goss', 'rf'
            n_estimators=250,              # More trees for better accuracy
            num_leaves=127,                # Aggressive: 2^7-1 leaves (complexity)
            max_depth=12,                  # Deep trees for capturing patterns
            learning_rate=0.0125,          # Moderate rate for stability
            subsample=0.85,                # High subsample for variance reduction
            subsample_freq=1,              # Enable bagging every iteration
            colsample_bytree=0.85,         # High feature fraction
            min_child_samples=20,          # Prevent overfitting on small groups
            min_child_weight=0.001,        # Min sum of hessian in leaf
            reg_alpha=0.1,                 # L1 regularization
            reg_lambda=1.0,                # L2 regularization
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True,           # Optimize for many features
        )
        self.model.fit(X_train, y_train)

    # ==========================================================================
    def predict(self, test_df):
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")

        test_processed = self.preprocess(test_df, is_train=False)
        X_test, _ = self.select_features(test_processed)
        X_test = X_test[self.feature_columns]

        predictions = self.model.predict(X_test)
        bad_mask = predictions <= 0

        if bad_mask.any():
            # global mean from valid preds; if none valid, use overall mean;
            # if still <=0, fallback to 1.0
            valid = predictions[~bad_mask]
            if valid.size:
                global_mean = float(valid.mean())
            else:
                global_mean = float(predictions.mean())
            if not np.isfinite(global_mean) or global_mean <= 0:
                global_mean = 1.0

            df_pred = test_processed.loc[:, ["category","merchant","state","city","zip","job","city_pop"]].copy()
            df_pred["pred"] = predictions
            groupings = [
                ["category", "merchant", "state", "city", "zip", "job", "city_pop"],
                ["category", "merchant", "state", "city", "job", "zip"],
                ["category", "merchant", "state", "city", "job"],
                ["category", "merchant", "state", "city"],
                ["category", "merchant", "state"],
                ["category", "merchant"],
                ["category"],
            ]
            for cols in groupings:
                cols = [c for c in cols if c in df_pred.columns]
                if not cols:
                    continue
                mask = df_pred["pred"] <= 0
                if not mask.any():
                    break
                grp = (
                    df_pred.loc[~mask, cols + ["pred"]]
                    .groupby(cols, dropna=False)["pred"]
                    .mean()
                    .reset_index()
                    .rename(columns={"pred": "pred_imputed"})
                )
                df_pred = df_pred.merge(grp, on=cols, how="left")
                impute_mask = mask & df_pred["pred_imputed"].notna()
                if impute_mask.any():
                    df_pred.loc[impute_mask, "pred"] = df_pred.loc[impute_mask, "pred_imputed"]
                df_pred.drop(columns=["pred_imputed"], inplace=True)

            # final fallback
            still_bad = df_pred["pred"] <= 0
            if still_bad.any():
                df_pred.loc[still_bad, "pred"] = global_mean

            predictions = df_pred["pred"].to_numpy()
        return np.clip(predictions, 1, None)

    # ==========================================================================
    def save_predictions(self, test_df, predictions):
        output = pd.DataFrame({
            'trans_num': test_df['trans_num'],
            'amt': predictions
        })
        output_file = f'z{self.student_id}_regression.csv'
        output.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")
        print(f"Regression training completed in {(time.time() - self.start_time):.2f}s")

#* =============================================================================
#* Part III: Classification - LightGBM
#* =============================================================================

class ClassificationFraudDetectionPipeline():
    def __init__(self, student_id):
        self.start_time = time.time()
        self.student_id = student_id
        self.model = None
        self.feature_columns = None

        # Saved stats from training
        self._city_freq_map = None
        self._cat_mean = None
        self._cat_median = None
        self._cat_std = None

        self._cat_hour_median  = None
        self._cat_hour_mean  = None

        self._global_mean = None
        self._global_std = None
        self._global_median = None
        self._global_mad = None

        self._merchant_mean = None
        self._job_mean = None
        self._age_median = None
    # ==========================================================================
    def preprocess(self, df, is_train=True):
        df = df.copy()

        # ----------------------------------------------------------------------
        # Parse datetime
        # ----------------------------------------------------------------------
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        s = df['trans_date_trans_time']

        # ----------------------------------------------------------------------
        # Time Features
        # ----------------------------------------------------------------------
        df['trans_hour'] = s.dt.hour.astype('float')
        df['trans_day'] = s.dt.day.astype('float')
        df['trans_month'] = s.dt.month.astype('float')
        df['trans_dayofweek'] = s.dt.dayofweek.astype('float')
        df['is_weekend'] = s.dt.dayofweek.isin([5, 6]).astype('int8')

        df['cos_hour'] = np.cos(2 * np.pi * (df['trans_hour'] / 24.0))
        df['hour_sin'] = np.sin(2 * np.pi * df['trans_hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['trans_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['trans_month'] / 12)

        df['weekend_9pm_6am'] = (
            ((df['trans_hour'] >= 22) | (df['trans_hour'] <= 3)) & (df['is_weekend'] == 1)
        ).astype('int8')

        df['weekday_9pm_6am'] = (
            ((df['trans_hour'] >= 22) | (df['trans_hour'] <= 3)) & (df['is_weekend'] == 0)
        ).astype('int8')

        df['is_daytime_7_20'] = df['trans_hour'].between(7, 20).astype('int8')
        df['is_peak_fraud_22_23'] = df['trans_hour'].between(22, 23).astype('int8')
        df['is_fraud_time'] = df['trans_hour'].isin([22, 23, 0, 1, 2, 3]).astype('int8')

        # Hour bin
        h = df['trans_hour'].astype('float')
        conds = [ (h >= 22) | (h <= 3), (h > 3) & (h <= 9), (h > 9) & (h <= 17),]
        choices = ['night', 'morning', 'day']
        hb = np.select(conds, choices, default='evening')
        df['hour_bin'] = pd.Categorical(hb, categories=['night','morning','day','evening'], ordered=True)
        df['hour_bin_code'] = df['hour_bin'].cat.codes

        # ----------------------------------------------------------------------
        # City Frequency
        # ----------------------------------------------------------------------
        if is_train:
            self._city_freq_map = df['city'].value_counts(normalize=True, dropna=False)
        df['city_freq'] = df['city'].map(self._city_freq_map)

        # Distance & city-pop features
        df['lat_diff'] = df['lat'] - df['merch_lat']
        df['long_diff'] = df['long'] - df['merch_long']
        df['distance_approx'] = np.sqrt(df['lat_diff']**2 + df['long_diff']**2)
        df['city_pop_log'] = np.log1p(df['city_pop'])

        # ----------------------------------------------------------------------
        # Category Stats
        # ----------------------------------------------------------------------
        if is_train:
            cat_stats = df.groupby('category')['amt'].agg(['mean', 'median', 'std'])
            self._cat_mean = cat_stats['mean']
            self._cat_median = cat_stats['median']
            self._cat_std = cat_stats['std']

        _c_mean = df['category'].map(self._cat_mean)
        _c_median = df['category'].map(self._cat_median)
        _c_std = df['category'].map(self._cat_std).replace(0, np.nan)

        df['amt_vs_cat_mean'] = (df['amt'] / _c_mean).replace([np.inf, -np.inf], np.nan)
        df['amt_vs_cat_median'] = (df['amt'] / _c_median).replace([np.inf, -np.inf], np.nan)
        df['amt_z_cat'] = (df['amt'] - _c_mean) / _c_std
        df['amt_outlier_cat'] = df['amt_z_cat'].abs().gt(3).astype('int8')

        # ----------------------------------------------------------------------
        # Category x hour bin stats
        # ----------------------------------------------------------------------
        if is_train:
            cat_hour_stats = df.groupby(['category', 'hour_bin'])['amt'].agg(['mean', 'median'])
            self._cat_hour_mean = cat_hour_stats['mean'].to_dict()
            self._cat_hour_median = cat_hour_stats['median'].to_dict()

        df['_cat_hour_key'] = list(zip(df['category'], df['hour_bin']))
        df['_catHr_mean'] = df['_cat_hour_key'].map(self._cat_hour_mean)
        df['_catHr_median'] = df['_cat_hour_key'].map(self._cat_hour_median)

        df['amt_vs_catHour_mean'] = (df['amt'] / df['_catHr_mean']).replace([np.inf, -np.inf], np.nan)
        df['amt_vs_catHour_median'] = (df['amt'] / df['_catHr_median']).replace([np.inf, -np.inf], np.nan)

        # Cleanup temporary columns
        df = df.drop(columns=['_cat_hour_key', '_catHr_mean', '_catHr_median'])

        # ----------------------------------------------------------------------
        # Global amount outliers
        # ----------------------------------------------------------------------
        if is_train:
            self._global_mean = df['amt'].mean()
            self._global_std = df['amt'].std()
            self._global_median = df['amt'].median()
            self._global_mad = (df['amt'] - self._global_median).abs().median()

        df['amt_z'] = (df['amt'] - self._global_mean) / (self._global_std if self._global_std else 1.0)
        df['amt_outlier'] = df['amt_z'].abs().gt(3).astype('int8')

        df['amt_z_robust'] = 0.6745 * (df['amt'] - self._global_median) / (self._global_mad if self._global_mad else 1.0)
        df['amt_z_robust'] = pd.to_numeric(df['amt_z_robust'], errors='coerce').replace([np.inf, -np.inf], np.nan)
        df['amt_outlier_robust'] = df['amt_z_robust'].abs().gt(3.5).astype('int8')

        # Extra interactions
        df['is_fraud_time_amt_z'] = df['is_fraud_time'] * df['amt_z']
        df['amt_z_cat_peak'] = df['amt_z_cat'] * df['is_peak_fraud_22_23']

        # ----------------------------------------------------------------------
        # Original Columns
        # ----------------------------------------------------------------------
        # Demographics first (as in original)
        df['gender_num'] = df['gender'].map({'F': 0, 'M': 1}).fillna(0.5).astype('float')

        # Age from DOB
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = (s - df['dob']).dt.days / 365.25
        if is_train:
            self._age_median = df['age'].median()
        df['age'] = df['age'].fillna(self._age_median if self._age_median else 40.0)

        # ----------------------------------------------------------------------
        # Merchant & Job
        # ----------------------------------------------------------------------
        if is_train:
            self._merchant_mean = df.groupby('merchant')['amt'].mean().to_dict()
        df['merchant_mean_amt'] = df['merchant'].map(self._merchant_mean)

        if is_train:
            self._job_mean = df.groupby('job')['amt'].mean().to_dict()
        df['job_mean_amt'] = df['job'].map(self._job_mean)

        return df

    # ==========================================================================
    def select_features(self, df):
        feature_columns = [
            # Amount outliers (strongest predictors)
            'amt_outlier_cat',
            'amt_outlier',
            'amt_outlier_robust',

            # Time-based flags
            'is_fraud_time',
            'is_peak_fraud_22_23',
            'weekday_9pm_6am',
            'weekend_9pm_6am',
            'is_daytime_7_20',

            # Time features
            'trans_hour',
            'cos_hour',
            'is_weekend',
            'hour_bin_code',

            # Amount features
            'amt_z',
            'amt_z_robust',
            'amt',

            # Category statistics
            'amt_vs_cat_mean',
            'amt_vs_cat_median',
            'amt_z_cat',
            'amt_vs_catHour_median',
            'amt_vs_catHour_mean',

            # Location
            'city_freq',

            # Extra aggregate
            'is_fraud_time_amt_z',
            'amt_z_cat_peak',

            # Engineered statistics
            'merchant_mean_amt',
            'job_mean_amt',

            # Time (extended)
            'trans_day',
            'trans_month',
            'trans_dayofweek',
            'hour_sin',
            'month_sin',
            'month_cos',
            'unix_time',

            # Location features
            'lat',
            'long',
            'merch_lat',
            'merch_long',
            'lat_diff',
            'long_diff',
            'distance_approx',
            'city_pop',
            'city_pop_log',

            # Demographics
            'gender_num',
            'age',

            # IDs
            'cc_num',
        ]
        return df[feature_columns], feature_columns

    # ==========================================================================
    def train(self, train_df):
        train_processed = self.preprocess(train_df, is_train=True)
        X_train, self.feature_columns = self.select_features(train_processed)
        y_train = train_processed['is_fraud']
        self.model = LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            n_estimators=600,
            num_leaves=127,
            max_depth=12,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            subsample_freq=1,
            min_child_samples=20,
            min_child_weight=0.001,
            reg_alpha=0.3,
            reg_lambda=2.0,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )
        self.model.fit(X_train, y_train)

    # ==========================================================================
    def predict(self, test_df):
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        test_processed = self.preprocess(test_df, is_train=False)
        X_test, _ = self.select_features(test_processed)
        X_test = X_test[self.feature_columns]  # Ensure identical order
        return self.model.predict(X_test)

    # ==========================================================================
    def save_predictions(self, test_df, predictions):
        output = pd.DataFrame({
            'trans_num': test_df['trans_num'],
            'is_fraud': predictions
        })
        output_file = f'z{self.student_id}_classification.csv'
        output.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")
        print(f"Classification training completed in {(time.time() - self.start_time):.2f}s")

#* =============================================================================
#* Main Execution
#* =============================================================================
def main():
    # 1. Validate command line arguments
    if len(argv) != 3:
        print("Usage: python3 z5494973.py <train_csv> <test_csv>")
        exit(1)

    train_file = argv[1]
    test_file = argv[2]

    # 2. Load data
    student_id = "5494973"
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION")
    print(f"Student ID: z{student_id}")
    print("="*70)
    print(f"\nLoading datasets...")
    print(f"  Train: {train_file}")
    print(f"  Test:  {test_file}")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    print(f"\nData loaded")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape:  {test_df.shape}")

    # 3. Part II: Regression - Predict Transaction Amount
    regression_pipeline = RegressionAmountPredictionPipeline(student_id)
    regression_pipeline.train(train_df)
    reg_predictions = regression_pipeline.predict(test_df)
    regression_pipeline.save_predictions(test_df, reg_predictions)

    # 4. Part III: Classification - Fraud Detection
    classification_pipeline = ClassificationFraudDetectionPipeline(student_id)
    classification_pipeline.train(train_df)
    cla_predictions = classification_pipeline.predict(test_df)
    classification_pipeline.save_predictions(test_df, cla_predictions)

    # 5. Training complete
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")

    # ==========================================================================
    # total_training_time = time.time() - total_start_time
    # print(f"Total Training completed in {total_training_time:.2f}s")
    # print(f"Total Training Time: {'✅ PASS' if total_training_time <= 120 else f'❌ FAIL - {total_training_time - 120:2f} away'}")

    # # check RMSE <= 140
    # if 'amt' in test_df.columns:
    #     rmse = np.sqrt(mean_squared_error(test_df['amt'], reg_predictions))
    #     print(f"\nRegression RMSE: {rmse:.2f}")
    #     print(f"    Target (<= 140): {'✅ PASS' if rmse <= 140 else f'❌ FAIL - {rmse - 140:.2f} away'}")

    #     # Mark estimate
    #     if rmse >= 180: mark = 0
    #     elif rmse <= 140: mark = 5
    #     else: mark = (1 - (rmse - 140)/(180 - 140)) * 5
    #     print(f"    Estimated regression mark (approx): {mark:.2f}/5")

    # # Calculate F1 score (if test labels available)
    # if 'is_fraud' in test_df.columns:
    #     y_test_true = test_df['is_fraud']
    #     f1 = f1_score(y_test_true, cla_predictions, average='macro')
    #     print(f"\nF1 Score (Macro): {f1:.4f}")
    #     print(f"    Target (>= 0.97): {'✅ PASS' if f1 >= 0.97 else f'❌ FAIL - {0.97 - f1:.4f} away'}")
    #     print(f"    Actual fraud:    {y_test_true.sum():,} ({y_test_true.mean()*100:.2f}%)")
    #     print(f"    Predicted fraud: {cla_predictions.sum():,} ({cla_predictions.mean()*100:.2f}%)")

    #     # Mark estimate
    #     if f1 <= 0.85: mark = 0
    #     elif f1 >= 97: mark = 5
    #     else: mark = ((f1 - 0.85) / (0.97 - 0.85)) * 5
    #     print(f"    Estimated classification mark (approx): {mark:.2f}/5")

#* =============================================================================
#* Entry Point
#* =============================================================================
if __name__ == "__main__":
    main()

