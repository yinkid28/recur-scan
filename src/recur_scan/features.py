from recur_scan.transactions import Transaction
from datetime import datetime
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




def get_features(transaction: Transaction, all_transactions: list[Transaction]) -> dict[str, float | int]:
    return {
         "n_transactions_same_amount": get_n_transactions_same_amount(transaction, all_transactions),
        "percent_transactions_same_amount": get_percent_transactions_same_amount(transaction, all_transactions),
        "avg_days_between_transactions": get_average_days_between_transactions(transaction, all_transactions),
        "time_regularity_score": get_time_regularity_score(transaction, all_transactions),
    }