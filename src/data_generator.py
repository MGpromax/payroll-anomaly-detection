"""
Generates fake payroll data for testing.
Creates realistic transactions with some anomalies injected.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional
import random


class PayrollDataGenerator:
    """
    Generates synthetic payroll transaction data with realistic patterns
    and injected anomalies for model testing.
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

        # Define company structure
        self.departments = ['Engineering', 'Sales', 'Marketing', 'Finance', 'HR', 'Operations']
        self.roles = {
            'Engineering': ['Junior Dev', 'Senior Dev', 'Tech Lead', 'Engineering Manager'],
            'Sales': ['Sales Rep', 'Senior Sales', 'Sales Manager', 'Regional Director'],
            'Marketing': ['Marketing Coord', 'Marketing Specialist', 'Marketing Manager'],
            'Finance': ['Analyst', 'Senior Analyst', 'Finance Manager', 'Controller'],
            'HR': ['HR Coord', 'HR Specialist', 'HR Manager'],
            'Operations': ['Operations Analyst', 'Operations Lead', 'Operations Manager']
        }

        # Salary bands by role level (annual, in thousands)
        self.salary_bands = {
            'Junior': (45, 65),
            'Senior': (70, 95),
            'Lead': (90, 120),
            'Manager': (100, 140),
            'Director': (130, 180),
            'Coord': (40, 55),
            'Specialist': (55, 75),
            'Analyst': (50, 70),
            'Rep': (45, 60),
            'Controller': (120, 160)
        }

    def _get_salary_band(self, role: str) -> Tuple[int, int]:
        """Get salary band based on role keywords"""
        for keyword, band in self.salary_bands.items():
            if keyword.lower() in role.lower():
                return band
        return (50, 80)  # Default band

    def _generate_employees(self, n_employees: int) -> pd.DataFrame:
        """Generate employee master data"""
        employees = []

        for emp_id in range(1, n_employees + 1):
            dept = random.choice(self.departments)
            role = random.choice(self.roles[dept])
            salary_band = self._get_salary_band(role)

            # Base salary within band
            base_salary = np.random.uniform(salary_band[0], salary_band[1]) * 1000

            # Tenure affects position in salary band
            tenure_years = np.random.exponential(3)  # Average 3 years tenure
            tenure_adjustment = min(tenure_years * 0.02, 0.15)  # Up to 15% for tenure
            base_salary *= (1 + tenure_adjustment)

            employees.append({
                'employee_id': f'EMP{emp_id:04d}',
                'department': dept,
                'role': role,
                'base_salary': round(base_salary, 2),
                'hire_date': datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1500)),
                'is_overtime_eligible': role not in ['Manager', 'Director', 'Controller', 'Tech Lead'],
                'max_weekly_ot_hours': 20 if 'Manager' not in role else 0
            })

        return pd.DataFrame(employees)

    def _generate_transactions(
        self,
        employees: pd.DataFrame,
        start_date: datetime,
        n_months: int
    ) -> pd.DataFrame:
        """Generate payroll transactions over time"""
        transactions = []

        for month_offset in range(n_months):
            pay_date = start_date + timedelta(days=30 * month_offset)

            for _, emp in employees.iterrows():
                # Regular salary (may have small variations for raises)
                if month_offset > 0 and month_offset % 12 == 0:
                    # Annual raise: 2-5%
                    raise_pct = np.random.uniform(0.02, 0.05)
                    emp['base_salary'] *= (1 + raise_pct)

                monthly_salary = emp['base_salary'] / 12

                # Overtime calculation
                ot_hours = 0
                ot_amount = 0
                if emp['is_overtime_eligible']:
                    # Most employees work 0-5 hours OT, some work more
                    if np.random.random() < 0.7:  # 70% work some overtime
                        ot_hours = np.random.exponential(3)  # Average 3 hours
                        ot_hours = min(ot_hours, emp['max_weekly_ot_hours'] * 4)  # Monthly cap
                        hourly_rate = (emp['base_salary'] / 52 / 40)
                        ot_amount = ot_hours * hourly_rate * 1.5  # Time and a half

                transactions.append({
                    'transaction_id': f'TXN{len(transactions)+1:08d}',
                    'employee_id': emp['employee_id'],
                    'department': emp['department'],
                    'role': emp['role'],
                    'transaction_date': pay_date,
                    'base_amount': round(monthly_salary, 2),
                    'overtime_hours': round(ot_hours, 1),
                    'overtime_amount': round(ot_amount, 2),
                    'total_amount': round(monthly_salary + ot_amount, 2),
                    'approver_id': f'MGR{hash(emp["department"]) % 20 + 1:03d}',
                    'processed_timestamp': pay_date + timedelta(hours=random.randint(8, 17))
                })

        return pd.DataFrame(transactions)

    def _inject_anomalies(
        self,
        transactions: pd.DataFrame,
        anomaly_rate: float = 0.02
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Inject realistic anomalies into the data.
        Returns modified transactions and ground truth labels.
        """
        df = transactions.copy()
        n_anomalies = int(len(df) * anomaly_rate)

        anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)
        anomaly_labels = pd.DataFrame({
            'transaction_id': df['transaction_id'],
            'is_anomaly': False,
            'anomaly_type': None
        })

        for idx in anomaly_indices:
            anomaly_type = random.choice([
                'salary_spike',
                'salary_manipulation',
                'excessive_overtime',
                'fake_overtime',
                'duplicate_payment'
            ])

            if anomaly_type == 'salary_spike':
                # Sudden unexplained salary increase (20-50%)
                multiplier = np.random.uniform(1.2, 1.5)
                df.loc[idx, 'base_amount'] *= multiplier
                df.loc[idx, 'total_amount'] = df.loc[idx, 'base_amount'] + df.loc[idx, 'overtime_amount']

            elif anomaly_type == 'salary_manipulation':
                # Small but persistent manipulation (5-15%)
                multiplier = np.random.uniform(1.05, 1.15)
                df.loc[idx, 'base_amount'] *= multiplier
                df.loc[idx, 'total_amount'] = df.loc[idx, 'base_amount'] + df.loc[idx, 'overtime_amount']

            elif anomaly_type == 'excessive_overtime':
                # Overtime beyond normal limits
                df.loc[idx, 'overtime_hours'] = np.random.uniform(60, 100)
                hourly_rate = df.loc[idx, 'base_amount'] * 12 / 52 / 40
                df.loc[idx, 'overtime_amount'] = df.loc[idx, 'overtime_hours'] * hourly_rate * 1.5
                df.loc[idx, 'total_amount'] = df.loc[idx, 'base_amount'] + df.loc[idx, 'overtime_amount']

            elif anomaly_type == 'fake_overtime':
                # Overtime for non-eligible employees or suspicious patterns
                df.loc[idx, 'overtime_hours'] = np.random.uniform(20, 40)
                hourly_rate = df.loc[idx, 'base_amount'] * 12 / 52 / 40
                df.loc[idx, 'overtime_amount'] = df.loc[idx, 'overtime_hours'] * hourly_rate * 1.5
                df.loc[idx, 'total_amount'] = df.loc[idx, 'base_amount'] + df.loc[idx, 'overtime_amount']

            elif anomaly_type == 'duplicate_payment':
                # Double payment
                df.loc[idx, 'total_amount'] *= 2

            anomaly_labels.loc[anomaly_labels['transaction_id'] == df.loc[idx, 'transaction_id'], 'is_anomaly'] = True
            anomaly_labels.loc[anomaly_labels['transaction_id'] == df.loc[idx, 'transaction_id'], 'anomaly_type'] = anomaly_type

        return df, anomaly_labels

    def generate(
        self,
        n_employees: int = 200,
        n_months: int = 24,
        anomaly_rate: float = 0.02,
        start_date: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate complete payroll dataset.

        Returns:
            - transactions: Payroll transactions
            - employees: Employee master data
            - ground_truth: Anomaly labels (for evaluation only)
        """
        if start_date is None:
            start_date = datetime(2022, 1, 1)

        print(f"Generating data for {n_employees} employees over {n_months} months...")

        employees = self._generate_employees(n_employees)
        transactions = self._generate_transactions(employees, start_date, n_months)
        transactions, ground_truth = self._inject_anomalies(transactions, anomaly_rate)

        print(f"Generated {len(transactions)} transactions")
        print(f"Injected {ground_truth['is_anomaly'].sum()} anomalies ({anomaly_rate*100:.1f}%)")

        return transactions, employees, ground_truth


def generate_sample_data(output_path: str = 'data/sample_payroll.csv'):
    """Generate and save sample data"""
    generator = PayrollDataGenerator(seed=42)
    transactions, employees, ground_truth = generator.generate(
        n_employees=200,
        n_months=24,
        anomaly_rate=0.02
    )

    # Save transactions
    transactions.to_csv(output_path, index=False)
    print(f"Saved transactions to {output_path}")

    # Save employee master (for reference)
    employees.to_csv(output_path.replace('.csv', '_employees.csv'), index=False)

    # Save ground truth (for evaluation only - would not exist in production)
    ground_truth.to_csv(output_path.replace('.csv', '_ground_truth.csv'), index=False)

    return transactions, employees, ground_truth


if __name__ == "__main__":
    generate_sample_data()
