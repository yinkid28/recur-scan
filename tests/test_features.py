# test features
import pytest
from datetime import datetime, timedelta
from recur_scan.features import (
    get_n_transactions_same_amount,
    get_percent_transactions_same_amount,
    parse_date,
    get_is_always_recurring,
)
from recur_scan.transactions import Transaction

@pytest.fixture
def transactions():
    """Fixture providing test transactions."""
    return [
        Transaction(id=1, user_id="user1", name="vendor1", amount=100, date="2024-01-01"),
        Transaction(id=2, user_id="user1", name="vendor1", amount=100, date="2024-01-02"),
        Transaction(id=3, user_id="user1", name="vendor1", amount=200, date="2024-01-03"),
        Transaction(id=4, user_id="user1", name="Netflix", amount=15.99, date="2024-01-05"),
    ]

def test_get_n_transactions_same_amount(transactions) -> None:
    """Test that get_n_transactions_same_amount returns the correct number of transactions with the same amount."""
    assert get_n_transactions_same_amount(transactions[0], transactions) == 2
    assert get_n_transactions_same_amount(transactions[2], transactions) == 1

def test_get_percent_transactions_same_amount(transactions) -> None:
    """
    Test that get_percent_transactions_same_amount returns correct percentage.
    Tests that the function calculates the right percentage of transactions with matching amounts.
    """
    assert pytest.approx(get_percent_transactions_same_amount(transactions[0], transactions)) == 0.5  # 2 out of 4 transactions have the same amount

# def test_parse_date() -> None:
#     """Test that parse_date correctly handles different date formats."""
#     # Test YYYY-MM-DD format
#     assert parse_date("2024-01-15") == datetime(2024, 1, 15)
    
#     # Test MM/DD/YYYY format
#     assert parse_date("01/15/2024") == datetime(2024, 1, 15)
    
#     # Test fallback
#     try:
#         result = parse_date("invalid-date")
#         assert result == datetime(1970, 1, 1)
#     except ValueError:
#         assert False, "parse_date should handle invalid dates gracefully"

@pytest.mark.parametrize(
    "transaction_name, expected_result",
    [
        # Test known recurring vendors (case-insensitive)
        ("Netflix", True),
        ("NETFLIX", True),
        ("netflix", True),
        ("Spotify", True),
        ("Adobe Creative Cloud", True),
        ("adobe creative cloud", True),
        
        # Test non-recurring vendors
        ("Walmart", False),
        ("Target", False),
        ("Restaurant XYZ", False),
        ("Gas Station", False),
        
        # Test edge cases
        ("", False),  # Empty vendor name
        ("Netflixx", False),  # Close but not exact match
        ("Google", False),  # Partial match but not in the list
    ],
)
def test_get_is_always_recurring(transaction_name, expected_result):
    """Test that get_is_always_recurring correctly identifies recurring vendors."""
    # Create a transaction with the given vendor name
    transaction = Transaction(id=1, user_id="user1", name=transaction_name, amount=100, date="2024-01-01")
    
    # Call the function and assert the result
    assert get_is_always_recurring(transaction) == expected_result