INTENT_KEYWORDS = {
    "mean": [
        "average",
        "mean",
        "avg"
    ],
    "sum": [
        "sum",
        "total",
        "add up",
        "aggregate"
    ],
    "count": [
        "count",
        "number of",
        "how many"
    ],
    "max": [
        "maximum",
        "max",
        "highest",
        "top"
    ],
    "min": [
        "minimum",
        "min",
        "lowest",
        "bottom"
    ],
    "filter": [
        "filter",
        "show",
        "only",
        "where",
        "with"
    ],
}

QUERIES = [
    "What's the average balance of married clients?",
    "How many clients defaulted on their loans?",
    "Filter clients who are unemployed and single.",
    "How many clients subscribed to the term deposit?",
    "Show me only clients with job management."
]

COLUMN_MATCH_THRESHOLD = 70

INTENT_HANDLERS = {
    'mean_balance': 'handle_mean_balance',
    'count_default': 'handle_count_default',
    'filter_clients': 'handle_filter_clients',
    'count_subscribed': 'handle_count_subscribed',
    'filter_job_marital': 'handle_filter_job_marital'
}