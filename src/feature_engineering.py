"""
Feature engineering for payroll anomaly detection.
Creates features that help identify salary manipulation and fake overtime.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats


class PayrollFeatureEngineer:
    """
    Takes raw payroll data and creates features for the ML model.
    Compares each transaction against department/role/personal baselines.
    """

    def __init__(self):
        self.dept_stats = {}
        self.role_stats = {}
        self.employee_history = {}
        self.fitted = False

    def fit(self, transactions: pd.DataFrame) -> 'PayrollFeatureEngineer':
        """
        Learn baseline statistics from historical transactions.

        This establishes what "normal" looks like for:
        - Department-level salary distributions
        - Role-level salary bands
        - Individual employee patterns
        """
        # Department-level statistics
        self.dept_stats = transactions.groupby('department').agg({
            'base_amount': ['mean', 'std', 'median'],
            'overtime_hours': ['mean', 'std', 'median', lambda x: x.quantile(0.95)],
            'overtime_amount': ['mean', 'std'],
            'total_amount': ['mean', 'std', 'median']
        }).to_dict()

        # Role-level statistics
        self.role_stats = transactions.groupby('role').agg({
            'base_amount': ['mean', 'std', 'min', 'max'],
            'overtime_hours': ['mean', 'std']
        }).to_dict()

        # Employee-level historical patterns
        for emp_id in transactions['employee_id'].unique():
            emp_data = transactions[transactions['employee_id'] == emp_id]
            self.employee_history[emp_id] = {
                'avg_salary': emp_data['base_amount'].mean(),
                'std_salary': emp_data['base_amount'].std(),
                'avg_ot_hours': emp_data['overtime_hours'].mean(),
                'std_ot_hours': emp_data['overtime_hours'].std(),
                'max_ot_hours': emp_data['overtime_hours'].max(),
                'ot_frequency': (emp_data['overtime_hours'] > 0).mean(),
                'transaction_count': len(emp_data)
            }

        self.fitted = True
        return self

    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features for anomaly detection.
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fit before transform")

        df = transactions.copy()
        features = pd.DataFrame(index=df.index)

        # === SALARY MANIPULATION FEATURES ===

        # 1. Salary vs Department Statistics
        features['salary_vs_dept_mean'] = df.apply(
            lambda x: self._z_score(
                x['base_amount'],
                self.dept_stats[('base_amount', 'mean')].get(x['department'], x['base_amount']),
                self.dept_stats[('base_amount', 'std')].get(x['department'], 1)
            ), axis=1
        )

        features['salary_vs_dept_median_ratio'] = df.apply(
            lambda x: x['base_amount'] / max(
                self.dept_stats[('base_amount', 'median')].get(x['department'], x['base_amount']),
                1
            ), axis=1
        )

        # 2. Salary vs Role Band
        features['salary_vs_role_mean'] = df.apply(
            lambda x: self._z_score(
                x['base_amount'],
                self.role_stats[('base_amount', 'mean')].get(x['role'], x['base_amount']),
                self.role_stats[('base_amount', 'std')].get(x['role'], 1)
            ), axis=1
        )

        # 3. Salary vs Personal History
        features['salary_vs_personal_avg'] = df.apply(
            lambda x: self._z_score(
                x['base_amount'],
                self.employee_history.get(x['employee_id'], {}).get('avg_salary', x['base_amount']),
                self.employee_history.get(x['employee_id'], {}).get('std_salary', 1) or 1
            ), axis=1
        )

        # 4. Salary Change Detection (month-over-month)
        features['salary_change_pct'] = self._calculate_salary_changes(df)

        # === OVERTIME FRAUD FEATURES ===

        # 5. Overtime vs Department Average
        features['ot_vs_dept_avg'] = df.apply(
            lambda x: self._z_score(
                x['overtime_hours'],
                self.dept_stats[('overtime_hours', 'mean')].get(x['department'], 0),
                self.dept_stats[('overtime_hours', 'std')].get(x['department'], 1) or 1
            ), axis=1
        )

        # 6. Overtime vs Personal Pattern
        features['ot_vs_personal_avg'] = df.apply(
            lambda x: self._z_score(
                x['overtime_hours'],
                self.employee_history.get(x['employee_id'], {}).get('avg_ot_hours', 0),
                self.employee_history.get(x['employee_id'], {}).get('std_ot_hours', 1) or 1
            ), axis=1
        )

        # 7. Overtime Exceeds Historical Max
        features['ot_exceeds_max'] = df.apply(
            lambda x: 1 if x['overtime_hours'] > self.employee_history.get(
                x['employee_id'], {}
            ).get('max_ot_hours', float('inf')) * 1.2 else 0, axis=1
        )

        # 8. Overtime Ratio (OT amount as % of total)
        features['ot_ratio'] = df['overtime_amount'] / df['total_amount'].replace(0, 1)

        # 9. Overtime Hours Percentile (within department)
        features['ot_percentile'] = df.groupby('department')['overtime_hours'].transform(
            lambda x: x.rank(pct=True)
        )

        # === COMBINED/BEHAVIORAL FEATURES ===

        # 10. Total Compensation Deviation
        features['total_vs_dept_mean'] = df.apply(
            lambda x: self._z_score(
                x['total_amount'],
                self.dept_stats[('total_amount', 'mean')].get(x['department'], x['total_amount']),
                self.dept_stats[('total_amount', 'std')].get(x['department'], 1)
            ), axis=1
        )

        # 11. Approver Pattern (encode as numeric for model)
        features['approver_encoded'] = pd.factorize(df['approver_id'])[0]

        # 12. Time-based features
        if 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            features['month'] = df['transaction_date'].dt.month
            features['is_year_end'] = (df['transaction_date'].dt.month == 12).astype(int)

        # 13. Composite Risk Score (simple heuristic)
        features['heuristic_risk'] = (
            (features['salary_vs_dept_mean'].abs() > 2).astype(int) +
            (features['ot_vs_dept_avg'].abs() > 2).astype(int) +
            (features['ot_exceeds_max'] == 1).astype(int) +
            (features['ot_ratio'] > 0.3).astype(int)
        ) / 4

        return features

    def fit_transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(transactions).transform(transactions)

    def _z_score(self, value: float, mean: float, std: float) -> float:
        """Calculate z-score with protection against division by zero"""
        if std == 0 or pd.isna(std):
            return 0
        return (value - mean) / std

    def _calculate_salary_changes(self, df: pd.DataFrame) -> pd.Series:
        """Calculate month-over-month salary change percentage"""
        df_sorted = df.sort_values(['employee_id', 'transaction_date'])

        changes = df_sorted.groupby('employee_id')['base_amount'].pct_change()

        # Fill NaN (first transaction for each employee) with 0
        changes = changes.fillna(0)

        return changes.reindex(df.index)

    def get_feature_names(self) -> List[str]:
        """Return list of feature names generated by transform"""
        return [
            'salary_vs_dept_mean',
            'salary_vs_dept_median_ratio',
            'salary_vs_role_mean',
            'salary_vs_personal_avg',
            'salary_change_pct',
            'ot_vs_dept_avg',
            'ot_vs_personal_avg',
            'ot_exceeds_max',
            'ot_ratio',
            'ot_percentile',
            'total_vs_dept_mean',
            'approver_encoded',
            'month',
            'is_year_end',
            'heuristic_risk'
        ]


def extract_features(
    transactions: pd.DataFrame,
    feature_engineer: Optional[PayrollFeatureEngineer] = None
) -> tuple:
    """
    Convenience function to extract features from transactions.

    Returns:
        features: DataFrame of engineered features
        feature_engineer: Fitted feature engineer (for reuse)
    """
    if feature_engineer is None:
        feature_engineer = PayrollFeatureEngineer()
        features = feature_engineer.fit_transform(transactions)
    else:
        features = feature_engineer.transform(transactions)

    return features, feature_engineer
