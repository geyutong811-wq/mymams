"""Cost monitoring module with actual cost accumulation and thread safety."""

import time
import threading
from typing import Dict, Any, List


class CostMonitor:
    """Track and accumulate costs from all API calls (thread-safe)."""

    def __init__(self):
        self.enabled: bool = True
        self.cost_log: List[Dict[str, Any]] = []
        self.total_cost: float = 0.0
        self._lock = threading.Lock()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def reset(self) -> None:
        with self._lock:
            self.cost_log.clear()
            self.total_cost = 0.0

    def add_cost(
        self,
        service: str,
        amount: float,
        success: bool = True,
        force_charge: bool = False,
        metadata: Dict[str, Any] = None
    ):
        """
        Add cost for a service.

        Args:
            service: Name of the service
            amount: Cost in USD
            success: Whether the operation succeeded. Failed operations are not charged by default.
            force_charge: If True, charge even if success=False (for timeout cases where request may have been processed).
            metadata: Additional info
        """
        if not self.enabled:
            return
        if not success and not force_charge:
            print(f"[CostMonitor] Skipping cost for failed operation: {service}")
            return
        with self._lock:
            entry = {
                "service": service,
                "amount": amount,
                "timestamp": time.time(),
                "success": success,
                "metadata": metadata or {}
            }
            self.cost_log.append(entry)
            self.total_cost += amount

    def get_total_cost(self) -> float:
        with self._lock:
            return self.total_cost

    def get_cost_report(self) -> Dict[str, Any]:
        with self._lock:
            breakdown = {}
            for entry in self.cost_log:
                service = entry["service"]
                breakdown[service] = breakdown.get(service, 0.0) + entry["amount"]
            return {
                "total": self.total_cost,
                "breakdown": breakdown,
                "currency": "USD",
                "transactions": self.cost_log.copy()
            }