# %%
import time
from functools import wraps
from typing import Any, Dict, Literal, Mapping, TypeVar

from elasticsearch import Elasticsearch

# from elasticsearch.client import create_ssl_context

T = TypeVar("T")

DEFAULT_RATE_LIMIT = 5


def rate_limited(func):
    @wraps(func)
    def wrapper(instance: Any, *args, **kwargs):
        # Fetch max_calls and period directly from instance attributes
        max_calls = instance.max_calls
        period = instance.period

        # If not already set, initialize calls and last_reset on the instance
        if not hasattr(instance, '_rate_limit_calls'):
            instance._rate_limit_calls = 0
            instance._rate_limit_last_reset = time.time()

        # Calculate time elapsed since last reset
        elapsed = time.time() - instance._rate_limit_last_reset

        # If elapsed time is greater than the period, reset the call count
        if elapsed > period:
            instance._rate_limit_calls = 0
            instance._rate_limit_last_reset = time.time()

        # Check if the call count has reached the maximum limit
        while instance._rate_limit_calls >= max_calls:
            # Instead of raising an exception, wait for the remaining time
            remaining_time = period - (time.time() - instance._rate_limit_last_reset)
            if remaining_time > 0:
                print(f"Rate limit exceeded! Waiting {round(remaining_time, 2)} sec...")
                time.sleep(remaining_time)

            # Now, reset the counters
            instance._rate_limit_calls = 0
            instance._rate_limit_last_reset = time.time()

        # Increment the call count
        instance._rate_limit_calls += 1

        # Call the original method
        return func(instance, *args, **kwargs)

    return wrapper


class SafeElasticsearch:
    def __init__(
        self,
        host: str,
        api_key: str,
        cert_path: str,
        index: str,
        max_calls: int = 5,
        period: float = 1.0,
    ) -> None:
        if max_calls / period > DEFAULT_RATE_LIMIT:
            raise Exception('CANNOT EXCEED 5 REQ/SEC RATE LIMIT WITHOUT APPROVAL')

        self.max_calls = max_calls
        self.period = period
        self._rate_limit_calls = 0
        self._rate_limit_last_reset = time.time()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"ApiKey {api_key}",
        }

        self.index = index

        import ssl

        self.client = Elasticsearch(
            hosts=[host],
            headers=headers,
            verify_certs=True,
            # ssl_show_warn=False,
            # ca_certs="../elastic.pem",
            ca_certs=cert_path,
        )

    @rate_limited
    def search(
        self,
        body: Dict[str, Any],
        explain: bool = False,
        min_score: "float | None" = None,
        pretty: "bool | None" = None,
        scroll: "str | Literal[-1, 0] | None" = '1m',
        size: int = 1500,
    ):

        return self.client.search(
            explain=explain,
            index=self.index,
            # min_score=min_score,
            pretty=pretty,
            body=body,
            scroll=scroll,
            size=size,
        )

    @rate_limited
    def scroll(self, scroll_id: str, scroll: "str | Literal[-1, 0] | None" = '1m'):
        return self.client.scroll(scroll_id=scroll_id, scroll=scroll)

    @rate_limited
    def clear_scroll(self, scroll_id: str):
        return self.client.clear_scroll(scroll_id=scroll_id)

    def close(self):
        self.client.close()
