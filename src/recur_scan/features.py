from recur_scan.transactions import Transaction
from datetime import datetime
import re
import statistics
from collections import Counter


def get_n_transactions_same_amount(transaction: Transaction, all_transactions: list[Transaction]) -> int:
    """Get the number of transactions in all_transactions with the same amount as transaction"""
    return len([t for t in all_transactions if t.amount == transaction.amount])


def get_percent_transactions_same_amount(transaction: Transaction, all_transactions: list[Transaction]) -> float:
    """Get the percentage of transactions in all_transactions with the same amount as transaction"""
    if not all_transactions:
        return 0.0
    n_same_amount = len([t for t in all_transactions if t.amount == transaction.amount])
    return n_same_amount / len(all_transactions)



#parse date 
def parse_date(date_str: str) -> datetime:
    """Parse date string into datetime object"""
    try:
        # Assuming date format is MM/DD/YYYY based on your sample
        return datetime.strptime(date_str, "%m/%d/%Y")
    except ValueError:
        # Fallback if format is different
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            # Return a default date if parsing fails
            return datetime(1970, 1, 1)


# get_average_days_between_transactions
def get_average_days_between_transactions(transaction: Transaction, all_transactions: list[Transaction]) -> float:
    """Calculate average days between transactions with same vendor"""
    same_vendor_txns = [t for t in all_transactions if t.name == transaction.name]
    if len(same_vendor_txns) <= 1:
        return 0.0
        
    # Sort by date
    try:
        sorted_txns = sorted(same_vendor_txns, key=lambda t: parse_date(t.date))
        
        # Calculate days between consecutive transactions
        days_between = []
        for i in range(1, len(sorted_txns)):
            days = (parse_date(sorted_txns[i].date) - parse_date(sorted_txns[i-1].date)).days
            days_between.append(days)
            
        return sum(days_between) / len(days_between) if days_between else 0.0
    except Exception:
        return 0.0

    

#get_time
def get_time_regularity_score(transaction: Transaction, all_transactions: list[Transaction]) -> float:
    """Calculate regularity of time intervals (lower std dev = more regular)"""
    same_vendor_txns = [t for t in all_transactions if t.name == transaction.name]
    if len(same_vendor_txns) <= 2:
        return 0.0
        
    try:
        # Sort by date
        sorted_txns = sorted(same_vendor_txns, key=lambda t: parse_date(t.date))
        
        # Calculate days between consecutive transactions
        days_between = []
        for i in range(1, len(sorted_txns)):
            days = (parse_date(sorted_txns[i].date) - parse_date(sorted_txns[i-1].date)).days
            days_between.append(days)
            
        # Lower standard deviation indicates more regular intervals
        if not days_between:
            return 0.0
            
        std_dev = statistics.stdev(days_between) if len(days_between) > 1 else 0.0
        # Convert to a score between 0 and 1 (1 = perfectly regular)
        return 1.0 / (1.0 + std_dev / 5.0)
    except Exception:
        return 0.0



def get_is_always_recurring(transaction: Transaction) -> bool:
    """Check if the transaction is always recurring because of the vendor name - check lowercase match"""
    always_recurring_vendors = {
        "google storage",
        "netflix",
        "hulu",
        "spotify",
        "amazon prime",
        "disney+",
        "apple music",
        "xbox game pass",
        "youtube premium",
        "adobe creative cloud",
    }
    return transaction.name.lower() in always_recurring_vendors



# New helper functions for date handling
def _get_days(date: str) -> int:
    """Get the number of days since the epoch of a transaction date."""
    try:
        date_obj = parse_date(date)
        return (date_obj - datetime(1970, 1, 1)).days
    except Exception:
        return 0
def _get_day(date: str) -> int:
    """Get the day of the month from a transaction date."""
    try:
        date_obj = parse_date(date)
        return date_obj.day
    except Exception:
        return 0
    

def get_n_transactions_same_day(transaction: Transaction, all_transactions: list[Transaction], n_days_off: int) -> int:
    """Get the number of transactions in all_transactions that are on the same day of the month as transaction"""
    transaction_day = _get_day(transaction.date)
    return len([t for t in all_transactions if abs(_get_day(t.date) - transaction_day) <= n_days_off])


def get_n_transactions_days_apart(
    transaction: Transaction,
    all_transactions: list[Transaction],
    n_days_apart: int,
    n_days_off: int,
) -> int:
    """
    Get the number of transactions in all_transactions that are within n_days_off of
    being n_days_apart from transaction
    """
    n_txs = 0
    transaction_days = _get_days(transaction.date)

    for t in all_transactions:
        t_days = _get_days(t.date)
        days_diff = abs(t_days - transaction_days)
        # skip if the difference is less than n_days_apart - n_days_off
        if days_diff < n_days_apart - n_days_off:
            continue

        # Check if the difference is close to any multiple of n_days_apart
        remainder = days_diff % n_days_apart

        if remainder <= n_days_off or (n_days_apart - remainder) <= n_days_off:
            n_txs += 1

    return n_txs






def get_features(transaction: Transaction, all_transactions: list[Transaction]) -> dict[str, float | int]:
    return {
        "n_transactions_same_amount": get_n_transactions_same_amount(transaction, all_transactions),
        "percent_transactions_same_amount": get_percent_transactions_same_amount(transaction, all_transactions),
        "avg_days_between_transactions": get_average_days_between_transactions(transaction, all_transactions),
        "time_regularity_score": get_time_regularity_score(transaction, all_transactions),
        "is_always_recurring": get_is_always_recurring(transaction),

        # Same day of month features
        "same_day_exact": get_n_transactions_same_day(transaction, all_transactions, 0),
        "same_day_off_by_1": get_n_transactions_same_day(transaction, all_transactions, 1),
        "same_day_off_by_2": get_n_transactions_same_day(transaction, all_transactions, 2),
    
        # Regular interval features
        "30_days_apart_exact": get_n_transactions_days_apart(transaction, all_transactions, 30, 0),
        "30_days_apart_off_by_1": get_n_transactions_days_apart(transaction, all_transactions, 30, 1),
        "14_days_apart_exact": get_n_transactions_days_apart(transaction, all_transactions, 14, 0),
        "14_days_apart_off_by_1": get_n_transactions_days_apart(transaction, all_transactions, 14, 1),
        "7_days_apart_exact": get_n_transactions_days_apart(transaction, all_transactions, 7, 0),
        "7_days_apart_off_by_1": get_n_transactions_days_apart(transaction, all_transactions, 7, 1),
    }

