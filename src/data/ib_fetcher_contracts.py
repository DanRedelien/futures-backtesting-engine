"""
Contract discovery and rolling helpers for IBFetcher.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ib_insync import Future


EXCHANGE_BY_SYMBOL = {
    "ES": "CME",
    "NQ": "CME",
    "RTY": "CME",
    "YM": "CBOT",
    "GC": "COMEX",
    "SI": "COMEX",
    "CL": "NYMEX",
    "NG": "NYMEX",
    "PL": "NYMEX",
    "ZC": "CBOT",
    "6E": "CME",
}


class IBFetcherContractsMixin:
    """Contract lookup helpers shared by the public IBFetcher."""

    def _get_exchange(self, symbol: str) -> str:
        """Returns the exchange code for one supported futures symbol."""
        return EXCHANGE_BY_SYMBOL.get(symbol, "CME")

    def _get_all_contracts(self, symbol: str) -> list[Future]:
        """
        Loads all available quarterly contracts for one symbol.

        Includes expired contracts so long backfills can roll correctly.
        """
        exchange = self._get_exchange(symbol)
        contract = Future(symbol, exchange=exchange)
        contract.includeExpired = True
        details = self.ib.reqContractDetails(contract)

        if not details:
            raise ValueError(f"Could not find contracts for {symbol}")

        quarterly_codes = ["H", "M", "U", "Z"]
        quarterly_contracts = [
            detail.contract
            for detail in details
            if len(detail.contract.localSymbol) >= 3
            and detail.contract.localSymbol[2] in quarterly_codes
        ]
        if not quarterly_contracts:
            quarterly_contracts = [detail.contract for detail in details]

        quarterly_contracts.sort(key=lambda item: item.lastTradeDateOrContractMonth)
        print(f"[INFO] Found {len(quarterly_contracts)} quarterly contracts for {symbol}")
        return quarterly_contracts

    def _get_contract_for_date(
        self,
        all_contracts: list[Future],
        target_date: datetime,
    ) -> Optional[Future]:
        """Returns the active contract for one historical date."""
        for contract in all_contracts:
            expiry_str = contract.lastTradeDateOrContractMonth
            if len(expiry_str) == 8:
                expiry = datetime.strptime(expiry_str, "%Y%m%d")
            else:
                expiry = datetime.strptime(expiry_str + "15", "%Y%m%d")
            if expiry > target_date:
                return contract
        return all_contracts[-1] if all_contracts else None
